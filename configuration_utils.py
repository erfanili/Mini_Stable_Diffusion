import functools


def register_to_config(init):
    
    
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        
        init_kwargs = {k:v for k,v in kwargs.items() if not k.startswith("_")}
    
        init(self, *args, **init_kwargs)
    
    
    return inner_init