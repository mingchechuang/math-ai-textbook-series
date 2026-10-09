<<<PATCH 01>>>
<<<OLD>>>
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
            
        self.d_model = d_model
<<<NEW>>>
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
            
        self.d_model = d_model
>>>
<<<END>>>