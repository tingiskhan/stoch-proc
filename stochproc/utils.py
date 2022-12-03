def lazy_property(property_name: str):
    """
    Helper method for evaluating lazy expressions.

    Args:
        property_name (str): name of the property to set on the dictionary.
    """

    def wrapper_outer(f):
        def wrapper_inner(self):
            prop = f(self)

            if callable(prop):
                # TODO: Perhaps verify that self is a dictionary...
                prop = self[property_name] = prop()
                        
            return prop

        return property(wrapper_inner, doc=f.__doc__)

    return wrapper_outer
