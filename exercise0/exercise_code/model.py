from exercise_code.test_class import TestClass
from exercise_code.test_class import TestModel
from exercise_code.data_utils import evaluate
from exercise_code.model_savers import save_test_model

test_class = TestClass()
test_class.write()

model = TestModel()
evaluate(model.return_score())

save_test_model(model)