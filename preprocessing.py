import numpy as np

'''
Step 1: Load the data and extract MTG samples
'''

# Use the metadata to find the number of MTG samples (column 20)
metadata = []
count = 0
with open("../data/metadata.csv", "r") as file:
    for index, line in enumerate(file):
        line = line.strip()
        line = line.split(',')
        metadata.append(line)
        if line[20] == "MTG":
            print(line)
            count += 1

print("Number of MTG samples: ", count)

with open("../output/metadata_loaded.csv", "w") as file:
    for line in metadata:
        file.writelines(line)

# # Use data file to retrieve read counts for MTG samples
# data = []
# with open("../data/matrix.csv", "r") as file:
#     for index, line in enumerate(file):
#         line = line.strip()
#         line = line.split(",")
#         meta = metadata[index]
#         region = meta[20]
#         if region == "MTG":
#             data.append(line)

# print("Shape of data: {r} x {c}".format(r=len(data), c=len(data[0])))

# data_arr = np.ones((len(data), len(data[0]) - 1))

# # Put data into array while skipping index col
# for index, row in enumerate(data):
#     row_arr = np.array(row[1:])
#     data_arr[index] = row_arr

# print("Shape of data array: {r} x {c}".format(
#     r=data_arr.shape[0], c=data_arr.shape[1]))

# np.savetxt("../output/data_loaded.csv", data_arr, delimiter=",", fmt="%.2f")

# print("Data saved.")
