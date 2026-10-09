<<<PATCH 17>>>
<<<OLD>>>
try:
    rope_with_start(x, -1)
    raise AssertionError("negative start was not rejected")
except ValueError:
    pass
<<<NEW>>>
caught = False
try:
    rope_with_start(x, -1)
except ValueError:
    caught = True
assert caught, "negative start was not rejected"
<<<END>>>