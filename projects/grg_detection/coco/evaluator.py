import numpy as np

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
        tp, fp, fn = self._gather_predictions()
        return {
            "accuracy": self._accuracy(tp, fp, fn),
            "precision": self._precision(tp, fp),
            "recall": self._recall(tp, fn)
        }

    def _gather_predictions(self):
        tp_list = []
        fp_list = []
        fn_list = []
        for image in self.annotations['images']:
            mask = self._get_mask_from_annotations(image)
            grg_components = self._extract_gt_components(image)
            all_components = self._extract_all_components(image)
            non_grg_components = self._remove_grg_from_all_components(all_components, grg_components)

            grg_components_in_mask = self._grg_components_are_in_mask(grg_components, mask)
            non_grg_components_in_mask = self._non_grg_components_are_in_mask(non_grg_components, mask)

            tp = self._tp(grg_components_in_mask, non_grg_components_in_mask)
            fp = self._fp(grg_components_in_mask, non_grg_components_in_mask)
            fn = self._fn(grg_components_in_mask, non_grg_components_in_mask)

            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        # Convert to numpy arrays for easier calculation of metrics
        # and also convert the bool values to integers (1 for True, 0 for False)
        # for metric calculations
        tp_list = np.array(tp_list).astype(int)
        fp_list = np.array(fp_list).astype(int)
        fn_list = np.array(fn_list).astype(int)
        return tp_list, fp_list, fn_list
    
    def _grg_components_are_in_mask(self, grg_components: list, mask: np.ndarray):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        for comp in grg_components:
            x, y = comp
            # Assuming mask is binary with 1 for predicted region and 0 for background
            if mask[int(y), int(x)] == 0:
                return False
        return True
    
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

    def _tp(self, grg_in_mask: bool, non_grg_in_mask: bool): # True Positives
        """Region uniquely encompasses all GRG components and no non-GRG components"""
        if grg_in_mask == True and non_grg_in_mask == False:
            return True
        return False

    def _fp(self, grg_in_mask: bool, non_grg_in_mask: bool): # False Positives
        """Region missing GRG components OR includes non-GRG components"""
        if grg_in_mask == False or non_grg_in_mask == True:
            return True
        return False

    def _fn(self, grg_in_mask: bool, non_grg_in_mask: bool): # False Negatives
        """No region covering GRG components (should be: not grg_in_mask)"""
        if grg_in_mask == False:
            return True
        return False
    