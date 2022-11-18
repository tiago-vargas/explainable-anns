import os


class TestMilpOutput:
	def test_milp_outputting_the_same_as_initially_outputted_by_levi(self):
		original_output = os.popen('cat console_outputs/original_milp_output').read()

		os.system('python3 milp.py > console_outputs/current_milp_output')
		current_output = os.popen('cat console_outputs/current_milp_output').read()

		assert current_output == original_output
