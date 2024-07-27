from typing import Dict

class CLIPImageProcessor():
    
    
    def __init__(
        self,
        de_resize: bool = True,
        size: Dict[str, int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 224}