import numpy as np


def truth_table_column(n_values, n_var, select=1):
    numbers = range(0, n_values)

    zeros = []
    for i in numbers:
        if i % (2 ** (n_var + 1)) == 0:
            for j in range(i, i + 2 ** n_var):
                if i >= n_values:
                    break
                zeros.append(i)
                i += 1

    if not select:
        return zeros
    else:
        ones = np.delete(numbers, zeros)
        return ones.tolist()


def is_pynn_object(obj, sim):
    is_population = isinstance(obj, sim.Population)
    is_population_view = isinstance(obj, sim.PopulationView)
    is_assembly = isinstance(obj, sim.Assembly)

    return is_population or is_population_view or is_assembly


def inverse_rcp_type(rcp_type):
    if rcp_type == "excitatory":
        return "inhibitory"
    elif rcp_type == "inhibitory":
        return "excitatory"
    else:
        raise ValueError("This receptor type is not supported")


def list_element(array, index_array):
    return array[index_array[0]]


def create_connections(ini_pop, end_pop, sim, conn, conn_all=True, rcp_type="excitatory", ini_pop_indexes=None,
                       end_pop_indexes=None):
    # Note: ini_pop and end_pop must be list or pynn object, and ini_pop_indexes and end_pop_indexes must be lists of
    # ints

    # Check ini_pop and end_pop object types
    ini_pop_islist = isinstance(ini_pop, list)
    end_pop_islist = isinstance(end_pop, list)

    # Functions to access elements of ini_pop and end_pop and calculation of ini_pop and end_pop sizes
    if not ini_pop_islist:
        ini_pop_function = sim.PopulationView
        ini_pop_size = ini_pop.size
    else:
        ini_pop_function = list_element
        ini_pop_size = len(ini_pop)

    if not end_pop_islist:
        end_pop_function = sim.PopulationView
        end_pop_size = end_pop.size
    else:
        end_pop_function = list_element
        end_pop_size = len(end_pop)

    # If ini_pop_indexes or end_pop_indexes are None, take all elements in ini_pop or end_pop respectively
    if ini_pop_indexes is None:
        ini_pop_indexes = range(ini_pop_size)

    if end_pop_indexes is None:
        end_pop_indexes = range(end_pop_size)

    # Length of ini_pop_indexes and end_pop_indexes arrays
    ini_pop_indexes_len = len(ini_pop_indexes)
    end_pop_indexes_len = len(end_pop_indexes)

    # Create connections
    if conn_all:  # AllToAll
        for i in ini_pop_indexes:
            for j in end_pop_indexes:
                sim.Projection(ini_pop_function(ini_pop, [i]), end_pop_function(end_pop, [j]),
                               sim.OneToOneConnector(), conn, receptor_type=rcp_type)
        created_connections = ini_pop_indexes_len * end_pop_indexes_len
    else:  # OneToOne
        if ini_pop_indexes_len != end_pop_indexes_len:
            raise ValueError("The number of selected elements of ini_pop and end_pop must be the same in OneToOne connections")

        for i in range(ini_pop_indexes_len):  # It could be the length of end_pop_indexes too
            sim.Projection(ini_pop_function(ini_pop, [ini_pop_indexes[i]]),
                           end_pop_function(end_pop, [end_pop_indexes[i]]),
                           sim.OneToOneConnector(), conn, receptor_type=rcp_type)
        created_connections = ini_pop_indexes_len

    return created_connections


def multiple_connect(function_name, population, components, conn, conn_all, rcp_type, ini_pop_indexes,
                     end_pop_indexes, component_indexes):

    created_connections = 0

    # ini_pop_indexes is an array of arrays
    if ini_pop_indexes is not None and ini_pop_indexes != [] and isinstance(ini_pop_indexes[0], list):
        # end_pop_indexes is an array of arrays
        if end_pop_indexes is not None and end_pop_indexes != [] and isinstance(end_pop_indexes[0], list):
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes[i],
                                                        end_pop_indexes=end_pop_indexes[i])
        # end_pop_indexes is None or an array
        else:
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes[i],
                                                        end_pop_indexes=end_pop_indexes)
    # ini_pop_indexes is None or an array
    else:
        # end_pop_indexes is an array of arrays
        if end_pop_indexes is not None and end_pop_indexes != [] and isinstance(end_pop_indexes[0], list):
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes,
                                                        end_pop_indexes=end_pop_indexes[i])
        # end_pop_indexes is None or an array
        else:
            for i in range(len(component_indexes)):
                connect_function = getattr(components[component_indexes[i]], function_name)
                created_connections += connect_function(population, conn, conn_all=conn_all, rcp_type=rcp_type,
                                                        ini_pop_indexes=ini_pop_indexes,
                                                        end_pop_indexes=end_pop_indexes)

    return created_connections


def flatten(array):
    if not array:
        return array
    if isinstance(array[0], list):
        return flatten(array[0]) + flatten(array[1:])
    return array[:1] + flatten(array[1:])

