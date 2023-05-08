import project3
def test_preprocess_text(self):
    input_text = "I'm gonna go to the store and buy some milk. Wanna come?"
    expected_output = "gon na go store buy milk wan na come"
    self.assertEqual(project3.preprocess_text(input_text), expected_output)