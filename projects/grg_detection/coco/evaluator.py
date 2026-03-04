import numpy as np
import copy

from .probe import COCOProbe


class GTEvaluator(COCOProbe):
    """
    If our predicted region uniquely encompasses the central
    coordinates of the (non-removed or reinserted) radiocomponents
    in accordance with the manual association, we have a true positive (TP).
    If the region does not encompass all of
    the radio components that belong together, we have a false positive (FP).
    If the region encompasses all the radio components that belong together,
    but also encompasses additional unrelated radio components, that also counts as a FP.
    If there is no region covering the central coordinate of the focussed radio component
    with a score surpassing the user-set threshold we have a false
    negative (FN). A true negative (TN) is the absence of a region
    where this is indeed warranted. True negatives should not appear
    in our data, as we only consider radio images centred on radio
    components with a signal-to-noise ratio surpassing five.
    """
    def __init__(self, annotations: dict, annotations_path: str):
        super().__init__(annotations, annotations_path)
    
    def evaluate(self):
        tp_mask, fp_mask, fn_mask, tp_bbox, fp_bbox, fn_bbox = self._gather_predictions()

        segm_precision = self._precision(tp_mask, fp_mask)
        segm_recall = self._recall(tp_mask, fn_mask)

        bbox_precision = self._precision(tp_bbox, fp_bbox)
        bbox_recall = self._recall(tp_bbox, fn_bbox)
        
        # Compute metrics
        results = {
            "segm_accuracy": self._accuracy(tp_mask, fp_mask, fn_mask),
            "segm_precision": segm_precision,
            "segm_recall": segm_recall,
            "segm_f1": self._f1(segm_precision, segm_recall),
            "bbox_accuracy": self._accuracy(tp_bbox, fp_bbox, fn_bbox),
            "bbox_precision": bbox_precision,
            "bbox_recall": bbox_recall,
            "bbox_f1": self._f1(bbox_precision, bbox_recall)
        }
        
        # Log the results in a nice table format
        print("===========================================================")
        print("Evaluation results against ground truth:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        print("===========================================================")
        return copy.deepcopy(results)

    def _gather_predictions(self):
        tp_mask_list = []
        fp_mask_list = []
        fn_mask_list = []
        tp_bbox_list = []
        fp_bbox_list = []
        fn_bbox_list = []
        for image in self.annotations['images']:
            image_id = image['id']
            image_metadata = image.get('metadata', {})

            if not image_metadata.get("grg_in_sample", False):
                # This is a negative sample, we can skip it for evaluation
                # against ground truth as it should not have any GRG components.
                continue
            
            # Get the predicted mask and bounding box for this image
            mask = self._get_mask_from_annotations(image)
            bbox = self._get_bbox_from_annotations(image)

            # Extract the GRG components and non-GRG components for this image
            grg_components = self._extract_gt_components(image_metadata)
            all_components = self._extract_all_components(image_metadata)
            
            # Convert grg_components and all_components from dict to list if they are dicts
            if isinstance(grg_components, dict):
                grg_components = list(grg_components.values())
            if isinstance(all_components, dict):
                all_components = list(all_components.values())

            non_grg_components = self._remove_grg_from_all_components(all_components, grg_components)

            # Check if the GRG components are in the predicted mask and bounding box
            all_grg_components_in_mask, some_grg_components_in_mask = self._grg_components_are_in_mask(
                grg_components, mask
            )
            all_grg_components_in_bbox, some_grg_components_in_bbox = self._grg_components_are_in_bbox(
                grg_components, bbox
            )
    
            # Check if the non-GRG components are in the predicted mask
            non_grg_components_in_mask = self._non_grg_components_are_in_mask(non_grg_components, mask)

            # Calculate TP, FP, FN for both mask and bbox evaluations
            tp_mask = self._tp(all_grg_components_in_mask, non_grg_components_in_mask)
            fp_mask = self._fp(all_grg_components_in_mask, some_grg_components_in_mask, non_grg_components_in_mask)
            fn_mask = self._fn(some_grg_components_in_mask)
            tp_bbox = self._tp(all_grg_components_in_bbox, non_grg_components_in_mask)
            fp_bbox = self._fp(all_grg_components_in_bbox, some_grg_components_in_bbox, non_grg_components_in_mask)
            fn_bbox = self._fn(some_grg_components_in_bbox)

            # Append results to lists for later aggregation
            tp_mask_list.append(tp_mask)
            fp_mask_list.append(fp_mask)
            fn_mask_list.append(fn_mask)
            tp_bbox_list.append(tp_bbox)
            fp_bbox_list.append(fp_bbox)
            fn_bbox_list.append(fn_bbox)

        # Convert to numpy arrays for easier calculation of metrics
        # and also convert the bool values to integers (1 for True, 0 for False)
        # for metric calculations
        tp_mask_list = np.array(tp_mask_list).astype(int)
        fp_mask_list = np.array(fp_mask_list).astype(int)
        fn_mask_list = np.array(fn_mask_list).astype(int)
        tp_bbox_list = np.array(tp_bbox_list).astype(int)
        fp_bbox_list = np.array(fp_bbox_list).astype(int)
        fn_bbox_list = np.array(fn_bbox_list).astype(int)
        
        return tp_mask_list, fp_mask_list, fn_mask_list, tp_bbox_list, fp_bbox_list, fn_bbox_list
    
    def _grg_components_are_in_mask(self, grg_components: list, mask: np.ndarray):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        all_in_mask = None
        some_in_mask = None
        for comp in grg_components:
            x, y = comp
            # Assuming mask is binary with 1 for predicted region and 0 for background
            if mask[int(y), int(x)] == 0:
                all_in_mask = False
                continue
            some_in_mask = True

        # If we never set some_in_mask to True,
        # it means none of the components are in the mask,
        # so we set it to False
        # If we never set all_in_mask to False,
        # it means all components are in the mask,
        # so we set it to True
        if all_in_mask == False and some_in_mask == None:
            some_in_mask = False
        if some_in_mask == True and all_in_mask == None:
            all_in_mask = True
        return all_in_mask, some_in_mask

    def _grg_components_are_in_bbox(self, grg_components: list, bbox: list):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        if not bbox:  # No bbox at all
            return False, False
    
        all_in_bbox = None
        some_in_bbox = None
        for comp in grg_components:
            x, y = comp
            if bbox:
                # Assuming bbox is a list of one bounding box
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if (x1 <= x <= x2 and y1 <= y <= y2):
                    # If the component is within the bounding box,
                    # we can consider it as covered by the prediction,
                    # even if it's not in the mask (for detection-only models)
                    some_in_bbox = True
                    continue
                all_in_bbox = False

        # If we never set some_in_bbox to True,
        # it means none of the components are in the bounding box,
        # so we set it to False
        # If we never set all_in_bbox to False,
        # it means all components are in the bounding box,
        # so we set it to True
        if all_in_bbox == False and some_in_bbox == None:
            some_in_bbox = False
        if some_in_bbox == True and all_in_bbox == None:
            all_in_bbox = True
        return all_in_bbox, some_in_bbox
    
    def _non_grg_components_are_in_mask(self, non_grg_components: list, mask: np.ndarray):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        for comp in non_grg_components:
            x, y = comp
            # Assuming mask is binary with 1 for predicted region and 0 for background
            if mask[int(y), int(x)] == 1:
                return True
        return False
    
    def _accuracy(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
        """Calculate accuracy from TP, FP, FN"""
        total = np.sum(tp) + np.sum(fp) + np.sum(fn)
        correct = np.sum(tp)
        return correct / total if total > 0 else 0.0

    def _precision(self, tp: np.ndarray, fp: np.ndarray):
        """Calculate precision from TP and FP"""
        tp_sum = np.sum(tp)
        fp_sum = np.sum(fp)
        return tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        
    def _recall(self, tp: np.ndarray, fn: np.ndarray):
        """Calculate recall from TP and FN"""
        tp_sum = np.sum(tp)
        fn_sum = np.sum(fn)
        return tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0

    def _f1(self, precision: float, recall: float):
        """Calculate F1 score from precision and recall"""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _tp(self, all_components_in: bool, non_grg_in_mask: bool):
        """Region uniquely encompasses all GRG components and no non-GRG components"""
        return all_components_in == True and non_grg_in_mask == False

    def _fp(self, all_components_in: bool, some_components_in: bool, non_grg_in_mask: bool):
        """Made a prediction that's wrong (partial or includes extras)"""
        # If no prediction at all, it's FN not FP
        if some_components_in == False:
            return False
        # If we detected something but it's imperfect
        return all_components_in == False or non_grg_in_mask == True

    def _fn(self, some_components_in: bool):
        """Failed to detect GRG components"""
        return some_components_in == False
