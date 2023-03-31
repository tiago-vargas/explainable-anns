from io import TextIOWrapper
import random


def generate_dataset(file: TextIOWrapper, number_of_rows):
	generate_header(file)
	generate_rows(file, number_of_rows)


def generate_header(file: TextIOWrapper):
	column_labels = ['A', 'target']
	file.write(str.join(',', column_labels) + '\n')


def generate_rows(file: TextIOWrapper, number_of_rows):
	for _ in range(number_of_rows):
		generate_data(file)


def generate_data(file: TextIOWrapper):
	n = random.random()

	if n >= 0.5:
		target = 1
	else:
		target = 0

	file.write(f'{n},{target}' + '\n')


with open('train.csv', 'w') as csv_file:
	generate_dataset(csv_file, 100)
