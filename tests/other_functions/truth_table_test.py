from neural_logic_gates.connection_functions import truth_table_column

for i in range(0, 3):
    test = truth_table_column(8, i, select=1)
    print(test)
