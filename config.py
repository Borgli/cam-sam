import inspect


class Config:
    number_of_images = 10
    image_size = (224, 224)

    @classmethod
    def add(cls, variable):
        # Get the caller's local variables
        caller_locals = inspect.currentframe().f_back.f_locals
        # Find the variable name in the caller's local variables
        var_name = None
        for name, val in caller_locals.items():
            if val is variable:
                var_name = name
                break
        if var_name is not None:
            setattr(cls, var_name, variable)
        else:
            raise ValueError("Variable not found in caller's local scope")
