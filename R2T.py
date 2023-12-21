import cplex
import mysql.connector
import numpy as np
from cplex import CplexError
from collections import defaultdict


class Query():
    # One primary relation/SJA count
    q12 = "select O_ORDERKEY as value from orders, lineitem  where o_orderkey = l_orderkey "

    # Two primary relation/SJA count
    q5 = (
        "select C_CUSTKEY, S_SUPPKEY  "
        "from customer, orders, lineitem, supplier, nation, region "
        "where c_custkey = o_custkey "
        "and l_orderkey = o_orderkey "
        "and l_suppkey = s_suppkey "
        "and c_nationkey = s_nationkey "
        "and s_nationkey = n_nationkey "
        "and n_regionkey = r_regionkey "
    )

    # One primary relation/SJA aggregation
    q18 = (
        "select c_custkey, l_quantity "
        "from customer, orders, lineitem "
        "where c_custkey = o_custkey "
        "and l_orderkey = o_orderkey "
    )

    # One primary relation/SPJA count
    q10 = (
        "select c_custkey, o_orderkey "
        "from customer, orders, lineitem, nation "
        "where c_custkey = o_custkey "
        "and l_orderkey = o_orderkey "
        "and c_nationkey = n_nationkey "
    )

    # SJA count situation
    query_type_sja_count = "SJA-COUNT"
    # SJA Aggregation
    query_type_sja_aggregation = "SJA-AGGREGATION"
    # SPJA Aggregation
    query_type_spja = "SPJA"


def get_query_tuple(query):
    mydb = mysql.connector.connect(
        host='localhost',
        user='****',
        password='****',
        database='tpch',
        auth_plugin='mysql_native_password'
    )
    cursor = mydb.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def lp_solver(var, objective_weight, up, lin_expr, res):
    try:
        prob = cplex.Cplex()
        prob.objective.set_sense(prob.objective.sense.maximize)

        prob.variables.add(obj=objective_weight, ub=up, names=var)
        prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr), rhs=res)

        start_time = prob.get_time()
        prob.solve()
        total_time = prob.get_time() - start_time
        print("Solution value  = {} when tau = {}".format(prob.solution.get_objective_value(), res[-1]))

    except CplexError as exc:
        print(exc)

    return prob.solution.get_objective_value(), total_time


def get_r2t_result(query_tuple, sql_type, PARAM, primary_relation_num=1):
    # get query results and  lp solver parameters
    query_result, var, objective_weight, lin_expr, up = get_lp_constraints(query_tuple, sql_type, primary_relation_num)

    # LP solver
    tau, beta, epsilon, data_scale = PARAM["tau"], PARAM["beta"], PARAM["epsilon"], PARAM["data_scale"]
    log_global_sensitivity = np.log2(data_scale)
    hat_query_result = 0
    total_time = 0
    for i in range(1, int(log_global_sensitivity) + 1):
        tau = 2 * tau
        # compute noise
        noice = np.random.laplace(loc=0, scale=log_global_sensitivity * tau / epsilon) \
                - log_global_sensitivity * np.log(log_global_sensitivity / beta) * tau / epsilon
        # get lp solver result
        res = []
        if sql_type == "SPJA":
            res = [0] * query_result + [tau] * (len(lin_expr) - query_result)
        else:
            res = [tau] * len(lin_expr)
        query_result_lp, time = lp_solver(var, objective_weight, up, lin_expr, res)
        # compute final result
        query_result_lp += noice
        total_time += time
        hat_query_result = max(hat_query_result, query_result_lp)
    print("The query result is: {}. The query result after R2T is: {}.".format(query_result, hat_query_result))
    relative_error = (query_result - hat_query_result) / query_result
    print("The relative error is: {:.3f}%. ".format(relative_error * 100))
    print("The time cost is: {}. ".format(total_time))
    return relative_error, total_time


def get_lp_constraints(query_tuple, sql_type, primary_relation_num):
    match sql_type:
        # SJA count situation
        case Query.query_type_sja_count:
            return get_lp_constraints_sja1(query_tuple)
        # SJA Aggregation
        case Query.query_type_sja_aggregation:
            return get_lp_constraints_sja2(query_tuple, primary_relation_num)
        # SPJA Aggregation
        case Query.query_type_spja:
            return get_lp_constraints_spja(query_tuple)
        case _:
            print("Unknown sql type! Try as sja count situation.")
            return get_lp_constraints_sja1(query_tuple)


def get_lp_constraints_sja1(query_tuple):
    count_result = len(query_tuple)
    var = ['u' + str(i) for i in range(1, count_result + 1)]
    indices = defaultdict(list)
    up = [1] * count_result
    for i, tuple in enumerate(query_tuple):
        indices[tuple].append(var[i])
    objective_weight = [1] * count_result
    lin_expr = []
    for i, var_list in enumerate(indices.values()):
        lin_expr.append([var_list, [1] * len(var_list)])
    return count_result, var, objective_weight, lin_expr, up


def get_lp_constraints_sja2(query_tuple, primary_relation_num):
    query_num = len(query_tuple)
    var = ['u' + str(i) for i in range(1, query_num + 1)]
    indices = defaultdict(list)
    up = []
    query_result = 0
    for i, tuple in enumerate(query_tuple):
        indices[tuple[0:primary_relation_num]].append(var[i])
        aggregation_value = float(tuple[primary_relation_num])
        up.append(aggregation_value)
        query_result += aggregation_value
    objective_weight = [1] * query_num
    lin_expr = []
    for i, var_list in enumerate(indices.values()):
        lin_expr.append([var_list, [1] * len(var_list)])
    return query_result, var, objective_weight, lin_expr, up


def get_lp_constraints_spja(query_tuple):
    query_num = len(query_tuple)
    indices_u = defaultdict(list)
    indices_v = defaultdict(list)
    u_var = ['u' + str(i) for i in range(1, query_num + 1)]
    for i, query_tuple in enumerate(query_tuple):
        indices_u[query_tuple[0]].append(u_var[i])
        indices_v[query_tuple[1]].append(u_var[i])
    projection_unique_list = list(indices_v.keys())
    projection_query_result = len(projection_unique_list)
    v_var = ['v' + str(i) for i in range(1, projection_query_result + 1)]
    var = v_var + u_var
    objective_weight = [1] * len(v_var) + [0] * len(u_var)

    up = [1] * (projection_query_result + query_num)
    lin_expr = []
    for i in range(0, projection_query_result):
        v_i_u_var = indices_v[projection_unique_list[i]]
        lin_expr.append([[v_var[i]] + v_i_u_var, [1] + [-1] * len(v_i_u_var)])
    for i, PR_KEY in enumerate(indices_u.values()):
        lin_expr.append([PR_KEY, [1] * len(PR_KEY)])

    return projection_query_result, var, objective_weight, lin_expr, up


PARAM = {
    "experiment_epoch": 1,
    "tau": 1,
    "beta": 0.1,
    "epsilon": 1,
    "data_scale": 5 * pow(10, 5)
}
query_tuple = get_query_tuple(Query.q18)
relative_error_list = []
time_list = []
for index in range(PARAM["experiment_epoch"]):
    relative_error, time = get_r2t_result(query_tuple, Query.query_type_sja_aggregation, PARAM)
    relative_error_list.append(relative_error * 100)
    time_list.append(time)
print("The relative error of 100 experiments:{}".format(relative_error_list))
print("The time cost of 100 experiments:{}".format(time_list))
relative_error_list.sort()
time_list.sort()
print("The relative error of 100 experiments after sort:{}".format(relative_error_list))
print("The time cost of 100 experiments after sort:{}".format(time_list))
print("The average relative error after removing the max and min 20:{:.4f}%".format(
    np.mean(np.array(relative_error_list)[20:80])))
print("The average time cost of {} experiment:{}".format(PARAM["experiment_epoch"], np.mean(np.array(time_list))))
