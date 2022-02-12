from typing import Union, List


class FeatureSelectionActor:
    def __init__(self, actor_name: str, select_best_n: int = None, keep_feature: Union[str, List[str]] = None,
                 ):
        self.actor_name = actor_name
        self.select_best_n = select_best_n
        self.feature_name = keep_feature
