import torch


# AP Calculation adapted from :
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
def calculate_AP(det_boxes, det_labels, det_scores, true_boxes, true_labels, iou_thresh, cpu):
    # All of them are list of tensors where one tensor stands for one image

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)
    # these are all lists of tensors of the same length, i.e. number of images
    n_classes = 2

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))

    if cpu:
        true_images = torch.LongTensor(true_images).to("cpu")  # (n_objects), total no. of objects across all images
        true_boxes = torch.cat(true_boxes, dim=0).to("cpu")  # (n_objects, 4)
        true_labels = torch.cat(true_labels, dim=0).to("cpu")  # (n_objects)
    else:
        true_images = torch.LongTensor(true_images)  # (n_objects), total no. of objects across all images
        true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
        true_labels = torch.cat(true_labels, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))

    if cpu:
        det_images = torch.LongTensor(det_images).to("cpu")  # (n_detections)
        det_boxes = torch.cat(det_boxes, dim=0).to("cpu")  # (n_detections, 4)
        det_labels = torch.cat(det_labels, dim=0).to("cpu")  # (n_detections)
        det_scores = torch.cat(det_scores, dim=0).to("cpu")  # (n_detections)
    else:
        det_images = torch.LongTensor(det_images)  # (n_detections)
        det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
        det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
        det_scores = torch.cat(det_scores, dim=0)  # (n_detections)
    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)

    # Extract only objects with this class
    true_class_images = true_images[true_labels == 1]  # (n_class_objects)
    true_class_boxes = true_boxes[true_labels == 1]  # (n_class_objects, 4)

    # Keep track of which true objects with this class have already been 'detected'
    # So far, none
    true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8)  # (n_class_objects)

    # Extract only detections with this class
    det_class_images = det_images[det_labels == 1]  # (n_class_detections)
    det_class_boxes = det_boxes[det_labels == 1]  # (n_class_detections, 4)
    det_class_scores = det_scores[det_labels == 1]  # (n_class_detections)
    n_class_detections = det_class_boxes.size(0)
    if n_class_detections == 0:
        return 0

    # Sort detections in decreasing order of confidence/scores
    det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
    det_class_images = det_class_images[sort_ind]  # (n_class_detections)
    det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

    # In the order of decreasing scores, check if true or false positive
    true_positives = torch.zeros((n_class_detections), dtype=torch.float)  # (n_class_detections)
    false_positives = torch.zeros((n_class_detections), dtype=torch.float)  # (n_class_detections)
    for d in range(n_class_detections):
        this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
        this_image = det_class_images[d]  # (), scalar

        # Find objects in the same image with this class, their difficulties, and whether they have been detected before
        object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)

        # If no such object in this image, then the detection is a false positive
        if object_boxes.size(0) == 0:
            false_positives[d] = 1
            continue

        # Find maximum overlap of this detection with objects in this image of this class
        overlaps = iou(this_detection_box, object_boxes, cpu)  # (1, n_class_objects_in_img)
        max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

        # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
        # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
        original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
        # We need 'original_ind' to update 'true_class_boxes_detected'

        # If the maximum overlap is greater than the threshold of 0.5, it's a match
        if max_overlap.item() > iou_thresh:
            # If this object has already not been detected, it's a true positive
            if true_class_boxes_detected[original_ind] == 0:
                true_positives[d] = 1
                true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
            # Otherwise, it's a false positive (since this object is already accounted for)
            else:
                false_positives[d] = 1
        # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
        else:
            false_positives[d] = 1

    # Compute cumulative precision and recall at each detection in the order of decreasing scores
    cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
    cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
    cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
    cumul_recall = cumul_true_positives / n_class_detections  # (n_class_detections)

    # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
    recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
    precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)  # (11)
    for i, t in enumerate(recall_thresholds):
        recalls_above_t = cumul_recall >= t
        if recalls_above_t.any():
            precisions[i] = cumul_precision[recalls_above_t].max()
        else:
            precisions[i] = 0.
    average_precisions[1 - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    return average_precisions


# Self Implemented
def iou(boxes1, boxes2, cpu):
    # Calculating the Intersection over Union between the ground truth boxes and the anchor boxes
    # First we get each coordinates as vectors separately
    boxes1 = boxes1.cuda()
    boxes2 = boxes2.cuda()
    ya1, xa1, ya2, xa2 = torch.chunk(boxes1, 4, dim=1)  # Valid Anchor Boxes
    yb1, xb1, yb2, xb2 = torch.chunk(boxes2[:, :4], 4, dim=1)  # Ground truth Boxes

    # Then we check the intersection by testing maximum/minimum
    inter_x1 = torch.max(xa1, torch.transpose(xb1, 0, 1))
    # This will result in (a n_anchor_box, 2) Tensor, which has the maximum for each ground truth box vs anchor box
    inter_y1 = torch.max(ya1, torch.transpose(yb1, 0, 1))
    inter_x2 = torch.min(xa2, torch.transpose(xb2, 0, 1))
    inter_y2 = torch.min(ya2, torch.transpose(yb2, 0, 1))

    blen = boxes2.shape[0]

    # Calculating the intersection
    if cpu:
        inter_area = torch.max((inter_y2 - inter_y1 + 1), torch.zeros(blen)) * torch.max(
            (inter_x2 - inter_x1 + 1), torch.zeros(blen))
    else:
        inter_area = torch.max((inter_y2- inter_y1 +1), torch.zeros(blen).cuda()) * torch.max((inter_x2 - inter_x1 + 1), torch.zeros(blen).cuda())
    # Calculating the Union
    boxes1_area = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)  # (8940, 1)
    boxes2_area = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)  # (2, 1)

    union_area = boxes1_area + torch.transpose(boxes2_area, 0, 1) - inter_area

    # Calculating the IoU
    iou = inter_area / union_area

    return iou
