def phi(i, r, s, elem="quadratic"):
    if elem == "linear":
        if i == 1:
            return 1 - r - s
        elif i == 2:
            return r
        elif i == 3:
            return s
    else:
        if i == 1:
            return (1 - r - s)*(1 - 2*r - 2*s)
        elif i == 2:
            return -r*(1 - 2*r)
        elif i == 3:
            return -s*(1 - 2*s)
        elif i == 4:
            return 4*r*(1 - r - s)
        elif i == 5:
            return 4*r*s
        elif i == 6:
            return 4*s*(1 - r - s)


def dphi_dr(i, r, s, elem="quadratic"):
    if elem == "linear":
        if i == 1:
            return -1
        elif i == 2:
            return 1
        elif i == 3:
            return 0
    else:
        if i == 1:
            return -1*(1 - 2*r - 2*s) - 2*(1 - r - s)
        elif i == 2:
            return -(1 - 2*r) + 2*r
        elif i == 3:
            return 0
        elif i == 4:
            return 4*(1 - r - s) - 4*r
        elif i == 5:
            return 4*s
        elif i == 6:
            return -4*s


def dphi_ds(i, r, s, elem="quadratic"):
    if elem == "linear":
        if i == 1:
            return -1
        elif i == 2:
            return 0
        elif i == 3:
            return -1
    else:
        if i == 1:
            return -(1 - 2*r - 2*s) - 2*(1 - r - s)
        elif i == 2:
            return 0
        elif i == 3:
            return -(1 - 2*s) + 2*s
        elif i == 4:
            return -4*r
        elif i == 5:
            return 4*r
        elif i == 6:
            return 4*(1 - r - s) - 4*s

