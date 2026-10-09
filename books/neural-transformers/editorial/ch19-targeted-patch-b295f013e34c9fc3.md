<<<PATCH 19>>>
<<<OLD>>>
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
<<<NEW>>>
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
<<<END>>>