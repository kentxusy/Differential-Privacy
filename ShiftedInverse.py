import math
import numpy as np

import mysql.connector


def get_query_tuple(query):
    mydb = mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456',
        database='tpch',
        auth_plugin='mysql_native_password'
    )
    cursor = mydb.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def k_selection(k, count, t, u, tau):
    f_k = [0] * (2 * tau + 1)
    j = 0
    # Iterate over the instance V
    for i in range(len(t)):
        # If sum of all but the largest j counters count(u) <= k - 1
        if sum(sorted(count.values())[:-j or None]) <= k - 1:
            f_k[j] = t[i]
        else:
            j += 1
            if j > 2 * tau:
                # Return f_k(V, j), for j in [2*tau]
                return f_k
            else:
                f_k[j] = t[i]

        # Increment count(u) for the user u contributing t(i)
        count[u[i]] += 1
    return f_k


def shift_inverse(f_k, PARAM):
    s = [-tau - 1] * PARAM["D"]
    for r in range(PARAM["D"]):
        if r == f_k[PARAM["tau"]]:
            s[r] = 0
        else:
            # Use binary search since f_k is in descending order
            j = binary_search(f_k, r)
            if j != -1:
                if 1 <= j <= tau:
                    s[r] = -tau + j - 1
                elif tau < j <= 2 * tau:
                    s[r] = tau - j

    # Sample r from [D] with probability proportional to exp(epsilon / 2 * s(V, r)), denoted by r_tilde
    p = np.array([np.exp(PARAM["epsilon"] / 2 * s[r]) for r in range(PARAM["D"])])
    p /= p.sum()
    r_tilde = np.random.choice(PARAM["D"], p=p)

    # Return M(V) = r_tilde
    return r_tilde

def binary_search(f_k, r):
    low, high = 1, 2 * tau
    while low <= high:
        mid = (low + high) // 2
        if f_k[mid] < r <= f_k[mid - 1]:
            return mid
        elif r > f_k[mid - 1]:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def get_query_result():
    sql = (
        "select c_custkey, l_quantity "
        "from customer, orders, lineitem "
        "where c_custkey = o_custkey "
        "and l_orderkey = o_orderkey "
    )
    result = np.array(get_query_tuple(sql))
    sorted_result = result[(-result[:, 1]).argsort()]
    u, t = np.hsplit(sorted_result, 2)
    u = u.flatten()
    t = t.flatten()
    t = t.astype(float)
    return u, t


def get_evaluation_error(t, r_tilde, percentile):
    r = np.percentile(t, percentile)
    relative_error = (r_tilde - r) / r
    print("The relative_error when percentile = {} is: {}".format(percentile, relative_error))
    rank_error = 0
    if relative_error != 0:
        r_tilde_index = np.where(t == r_tilde)
        rank_error = r_tilde_index[0][0] - math.ceil((100 - percentile) / 100 * len(t))
    print("The rank error when percentile = {} is: {}".format(percentile, rank_error))


PARAM = {
    "beta": 1 / 3,
    "epsilon": 1,
    "D": pow(10, 5)
}
tau = math.ceil(2 * PARAM["epsilon"] * math.log((PARAM["D"] + 1) / PARAM["beta"]))
PARAM["tau"] = tau
user, query_tuple = get_query_result()
count = {u: 0 for u in np.unique(user)}
# k_selection when maximum
# f_k = k_selection(k=1, count=count, t=query_tuple, u=user, tau=PARAM["tau"])
# k_selection when percentile = 25
f_k = k_selection(k=math.ceil(50 / 100 * len(query_tuple)), count=count, t=query_tuple, u=user, tau=PARAM["tau"])
# k_selection when percentile = 75
# f_k = k_selection(k=math.ceil(25 / 100 * len(query_tuple)), count=count, t=query_tuple, u=user, tau=PARAM["tau"])
# get r_tilde
r_tilde = shift_inverse(f_k=f_k, PARAM=PARAM)
print("The result r_tidle is:{}".format(r_tilde))
# evaluate rank_error and relative error
get_evaluation_error(t=query_tuple, r_tilde=r_tilde, percentile=50)
print("test")
