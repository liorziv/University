import csv

researcher_id = 1
name_to_id = {}

with open("tests/all_edges.edges", "rb") as infile:

	with open("tests/numbered.edges", "wb") as outfile:
		reader = csv.reader(infile, delimiter="\t")
		writer = csv.writer(outfile, delimiter="\t")

		for row in reader:

			src, dest, weight = tuple(row)

			if src not in name_to_id:
				name_to_id[src] = researcher_id
				researcher_id += 1

			if dest not in name_to_id:
				name_to_id[dest] = researcher_id
				researcher_id += 1

			writer.writerow([name_to_id[src], name_to_id[dest], weight])

with open("tests/researcher_id.csv", "wb") as fl:
	writer = csv.writer(fl, delimiter="\t")
	for key, value in name_to_id.items():
		writer.writerow([value, key])

with open("tests/researcher_name.csv", "wb") as fl:
	writer = csv.writer(fl, delimiter="\t")
	for key, value in name_to_id.items():
		writer.writerow([key, value])