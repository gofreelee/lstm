import csv


csv_file =  open("8gemv_sum_20cells.csv")
time_data = [n for n in range(119)]
for i in csv_file:
    row_data = i.split(',')
    if len(row_data) == 9:
        time_ = float(row_data[3])
        func_type = row_data[6].split('_')[1]
        func_number = row_data[6].split('_')[2].split('(')[0]
        if func_type == 'compute':
            time_data[int(func_number)] = time_
for i in time_data:
    print(i)
