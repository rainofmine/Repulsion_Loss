import numpy as np
import torch
import torch.nn as nn
import random


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def IoG(box_a, box_b):

    inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
    inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
    inter_xmax = torch.min(box_a[:, 2], box_b[:, 2])
    inter_ymax = torch.min(box_a[:, 3], box_b[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    return I / G


def smooth_ln(x, smooth):
    return torch.where(
        torch.le(x, smooth),
        -torch.log(1 - x),
        ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
    )


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations, ignores):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        RepGT_losses = []
        RepBox_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                RepGT_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            ignore = ignores[j, :, :]
            ignore = ignore[ignore[:, 4] != -1]
            if ignore.shape[0] > 0:
                iou_igno = calc_iou(anchor, ignore)
                iou_igno_max, iou_igno_argmax = torch.max(iou_igno, dim=1)
                index_igno = torch.lt(iou_igno_max, 0.5)
                anchor_keep = anchor[index_igno, :]
                classification = classification[index_igno, :]
                regression = regression[index_igno, :]
                anchor_widths_keep = anchor_widths[index_igno]
                anchor_heights_keep = anchor_heights[index_igno]
                anchor_ctr_x_keep = anchor_ctr_x[index_igno]
                anchor_ctr_y_keep = anchor_ctr_y[index_igno]
            else:
                anchor_keep = anchor
                anchor_widths_keep = anchor_widths
                anchor_heights_keep = anchor_heights
                anchor_ctr_x_keep = anchor_ctr_x
                anchor_ctr_y_keep = anchor_ctr_y

            IoU = calc_iou(anchor_keep, bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]   #which gt the anchor matches

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]    #num_pos_anchors * 5

                anchor_widths_pi = anchor_widths_keep[positive_indices]
                anchor_heights_pi = anchor_heights_keep[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x_keep[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y_keep[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                negative_indices = 1 - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_loss = regression_loss.mean()
                regression_losses.append(regression_loss)


                # predict regression to boxes that are positive
                if bbox_annotation.shape[0] == 1:
                    RepGT_losses.append(torch.tensor(0).float().cuda())
                    # RepBox_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_pos = regression[positive_indices, :]
                    regression_pos_dx = regression_pos[:, 0]
                    regression_pos_dy = regression_pos[:, 1]
                    regression_pos_dw = regression_pos[:, 2]
                    regression_pos_dh = regression_pos[:, 3]
                    predict_w = torch.exp(regression_pos_dw) * anchor_widths_pi
                    predict_h = torch.exp(regression_pos_dh) * anchor_heights_pi
                    predict_x = regression_pos_dx * anchor_widths_pi + anchor_ctr_x_pi
                    predict_y = regression_pos_dy * anchor_heights_pi + anchor_ctr_y_pi
                    predict_xmin = predict_x - 0.5 * predict_w
                    predict_ymin = predict_y - 0.5 * predict_h
                    predict_xmax = predict_x + 0.5 * predict_w
                    predict_ymax = predict_y + 0.5 * predict_h
                    predict_boxes = torch.stack((predict_xmin, predict_ymin, predict_xmax, predict_ymax)).t()

                    # add RepGT_losses
                    IoU_pos = IoU[positive_indices, :]
                    IoU_max_keep, IoU_argmax_keep = torch.max(IoU_pos, dim=1, keepdim=True)  # num_anchors x 1
                    for idx in range(IoU_argmax_keep.shape[0]):
                        IoU_pos[idx, IoU_argmax_keep[idx]] = -1
                    IoU_sec, IoU_argsec = torch.max(IoU_pos, dim=1)

                    assigned_annotations_sec = bbox_annotation[IoU_argsec, :]    # which gt the anchor iou second num_anchors * 5

                    IoG_to_minimize = IoG(assigned_annotations_sec, predict_boxes)
                    RepGT_loss = smooth_ln(IoG_to_minimize, 0.5)
                    RepGT_loss = RepGT_loss.mean()
                    RepGT_losses.append(RepGT_loss)

                    # add PepBox losses
                    IoU_argmax_pos = IoU_argmax[positive_indices].float()
                    IoU_argmax_pos = IoU_argmax_pos.unsqueeze(0).t()
                    predict_boxes = torch.cat([predict_boxes, IoU_argmax_pos], dim=1)
                    predict_boxes_np = predict_boxes.detach().cpu().numpy()
                    num_gt = bbox_annotation.shape[0]
                    predict_boxes_sampled = []
                    for id in range(num_gt):
                        index = np.where(predict_boxes_np[:, 4]==id)[0]
                        if index.shape[0]:
                            idx = random.choice(range(index.shape[0]))
                            predict_boxes_sampled.append(predict_boxes[index[idx], :4])
                    predict_boxes_sampled = torch.stack(predict_boxes_sampled)
                    iou_repbox = calc_iou(predict_boxes_sampled, predict_boxes_sampled)
                    mask = torch.lt(iou_repbox, 1.).float()
                    iou_repbox = iou_repbox * mask
                    RepBox_loss = smooth_ln(iou_repbox, 0.5)
                    RepBox_loss = RepBox_loss.sum() / torch.clamp(torch.sum(torch.gt(iou_repbox, 0)).float(), min=1.0)
                    RepBox_losses.append(RepBox_loss)

            else:
                regression_losses.append(torch.tensor(0).float().cuda())
                RepGT_losses.append(torch.tensor(0).float().cuda())
                RepBox_losses.append(torch.tensor(0).float().cuda())
        
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True), \
               torch.stack(RepGT_losses).mean(dim=0, keepdim=True), \
               torch.stack(RepBox_losses).mean(dim=0, keepdim=True)


