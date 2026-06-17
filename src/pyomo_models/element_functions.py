import pyomo.environ as pyo


def add_level_addition_elements(model, m_scen):
    ## Parameters

    m_scen.sound_power_level_out_intermediate = pyo.Var(
        model.E,
        model.intervals,
        within=pyo.Reals,
        bounds=(-60, model.max_sound_power_level),
    )

    m_scen.level_increase1 = pyo.Var(
        model.E,
        model.intervals,
        within=pyo.NonNegativeReals,
        bounds=(0, model.max_sound_power_level),
    )

    m_scen.level_increase2 = pyo.Var(
        model.E,
        model.intervals,
        within=pyo.NonNegativeReals,
        bounds=(0, model.max_sound_power_level),
    )

    ## Variables

    @m_scen.Expression(model.E, model.intervals, doc="")
    def dampened_sound_power_level_in(m_scen, i, j, f):
        return (
            m_scen.sound_power_level[i, f]
            - m_scen.performance_curve_sound_power_level_dampening[i, j, f]
        )

    @m_scen.Expression(model.level_add_polyhedral_coeff_set, model.E, model.intervals, doc="")
    def linear_diff1(m_scen, t, i, j, f):
        return (
            model.level_add_polyhedral_approx_slope[t]
            * (
                m_scen.dampened_sound_power_level_in[i, j, f]
                - m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
            )
            + model.level_add_polyhedral_approx_y_intercept[t]
        )

    @m_scen.Expression(model.level_add_polyhedral_coeff_set, model.E, model.intervals, doc="")
    def linear_diff2(m_scen, t, i, j, f):
        return (
            model.level_add_polyhedral_approx_slope[t]
            * (
                -m_scen.dampened_sound_power_level_in[i, j, f]
                + m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
            )
            + model.level_add_polyhedral_approx_y_intercept[t]
        )

    @m_scen.Constraint(model.level_add_polyhedral_coeff_set, model.E, model.intervals)
    def level_difference1(m_scen, t, i, j, f):
        return m_scen.level_increase1[i, j, f] >= m_scen.linear_diff1[t, i, j, f]

    @m_scen.Constraint(model.level_add_polyhedral_coeff_set, model.E, model.intervals)
    def level_difference2(m_scen, t, i, j, f):
        return m_scen.level_increase2[i, j, f] >= m_scen.linear_diff2[t, i, j, f]

    @m_scen.Constraint(model.E, model.intervals, doc="")
    def sound_power_level_out_with_linear_polyhedrals1(m_scen, i, j, f):
        return (
            m_scen.sound_power_level_out_intermediate[i, j, f]
            >= m_scen.level_increase1[i, j, f]
            + m_scen.dampened_sound_power_level_in[i, j, f]
        )

    @m_scen.Constraint(model.E, model.intervals, doc="")
    def sound_power_level_out_with_linear_polyhedrals2(m_scen, i, j, f):
        return (
            m_scen.sound_power_level_out_intermediate[i, j, f]
            >= m_scen.level_increase2[i, j, f]
            + m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
        )

    return m_scen


def level_add_room(model, m_scen):
    """
    Level adding all elements of an indexed expression. The final value is the last block.w value.
    """
    expression = m_scen.sound_power_level_in_A_weighted

    # Variables

    m_scen.room_level_increase1 = pyo.Var(
        model.V_room,
        model.room_level_add_set,
        within=pyo.NonNegativeReals,
        bounds=(0, model.max_sound_power_level),
    )

    m_scen.room_level_increase2 = pyo.Var(
        model.V_room,
        model.room_level_add_set,
        within=pyo.NonNegativeReals,
        bounds=(0, model.max_sound_power_level),
    )

    m_scen.sound_power_level_in_level_addition_room = pyo.Var(
        model.V_room,
        model.room_level_add_set,
        within=pyo.Reals,
        bounds=(-20, model.max_sound_power_level),
        doc="Helper variable used for tournament style level addition",
    )

    # Expression

    @m_scen.Expression(model.V_room, model.room_level_add_set)
    def room_difference(m_scen, v, i):
        if i == 1:
            return expression[v, 1] - expression[v, 2]
        return m_scen.sound_power_level_in_level_addition_room[v, i - 1] - expression[v, i + 1]

    @m_scen.Expression(model.V_room, model.level_add_polyhedral_coeff_set, model.room_level_add_set, doc="")
    def room_linear_diff1(m_scen, v, t, i):
        return (
            model.level_add_polyhedral_approx_slope[t] * (m_scen.room_difference[v, i])
            + model.level_add_polyhedral_approx_y_intercept[t]
        )

    @m_scen.Expression(model.V_room, model.level_add_polyhedral_coeff_set, model.room_level_add_set, doc="")
    def room_linear_diff2(m_scen, v, t, i):
        return (
            model.level_add_polyhedral_approx_slope[t] * (-m_scen.room_difference[v, i])
            + model.level_add_polyhedral_approx_y_intercept[t]
        )

    # Constraint

    @m_scen.Constraint(model.V_room, model.level_add_polyhedral_coeff_set, model.room_level_add_set)
    def room_level_difference1(m_scen, v, t, i):
        return m_scen.room_level_increase1[v, i] >= m_scen.room_linear_diff1[v, t, i]

    @m_scen.Constraint(model.V_room, model.level_add_polyhedral_coeff_set, model.room_level_add_set)
    def room_level_difference2(m_scen, v, t, i):
        return m_scen.room_level_increase2[v, i] >= m_scen.room_linear_diff2[v, t, i]

    @m_scen.Constraint(model.V_room, model.room_level_add_set, doc="")
    def room_sound_power_level_out_with_linear_polyhedrals1(m_scen, v, i):
        if i == 1:
            return m_scen.sound_power_level_in_level_addition_room[v, 1] >= expression[v, 1] + m_scen.room_level_increase1[v, 1]
        return m_scen.sound_power_level_in_level_addition_room[v, i] >= m_scen.sound_power_level_in_level_addition_room[v, i - 1] + m_scen.room_level_increase1[v, i]

    @m_scen.Constraint(model.V_room, model.room_level_add_set, doc="")
    def room_sound_power_level_out_with_linear_polyhedrals2(m_scen, v, i):
        if i == 1:
            return m_scen.sound_power_level_in_level_addition_room[v, 1] >= expression[v, 2] + m_scen.room_level_increase2[v, 1]
        return m_scen.sound_power_level_in_level_addition_room[v, i] >= expression[v, i + 1] + m_scen.room_level_increase2[v, i]

    return m_scen


def level_add_multiple_multiindex(
    model,
    m_scen,
):
    """
    Level adding over one of the indices of a multi-indexed expression while keeping the other index. The final value is the last block.w value.
    """

    types = lambda i,j,x: model.fan_types_on_edge[i,j].at(x)

    # n_expr = int(len(m_scen.sound_power_level_identical_fans) / len(model.E_fan_station) / len(model.intervals))

    def level_add_set_multi_init(m_scen):
        return [(i,j,x) for (i,j) in model.E_fan_station_multi for x in range(1,max(len(model.fan_types_on_edge[i,j]),2))]

    def level_add_set_single_init(m_scen):
        return [(i,j,1) for (i,j) in model.E_fan_station_single]

    m_scen.level_add_set_multi = pyo.Set(
        initialize=level_add_set_multi_init, doc="set used for tournament style level addition"
    )

    m_scen.level_add_set_single = pyo.Set(
        initialize=level_add_set_single_init, doc="set used for tournament style level addition"
    )

    m_scen.polyhedral_coeff_set_multi = pyo.RangeSet(
        3, doc="set used for the coefficients of the polyhedral approximation"
    )

    m_scen.polyhedral_approx_slope_multi = pyo.Param(
        m_scen.polyhedral_coeff_set_multi,
        initialize={1: -0.415, 2: -0.219, 3: -0.066},
        doc="slope of the linear polyhedral approximation of the level rise",
    )

    m_scen.polyhedral_approx_y_intercept_multi = pyo.Param(
        m_scen.polyhedral_coeff_set_multi,
        initialize={1: 2.943, 2: 2.288, 3: 1.056},
        doc="y intercept of the linear polyhedral approximation of the level rise",
    )

    # Variables

    m_scen.level_increase1_multi = pyo.Var(
        m_scen.level_add_set_multi,
        model.intervals,
        within=pyo.NonNegativeReals,
        bounds=(0, model.max_sound_power_level),
    )

    m_scen.level_increase2_multi = pyo.Var(
        m_scen.level_add_set_multi,
        model.intervals,
        within=pyo.NonNegativeReals,
        bounds=(0, model.max_sound_power_level),
    )

    m_scen.sound_power_level_in_level_addition = pyo.Var(
        m_scen.level_add_set_multi | m_scen.level_add_set_single,
        model.intervals,
        within=pyo.Reals,
        bounds=(-20, model.max_sound_power_level),
        doc="Helper variable used for tournament style level addition",
    )

    # Expression

    @m_scen.Expression(m_scen.level_add_set_multi, model.intervals)
    def difference_multi(m_scen, i, j, ind, f):
        if ind == 1:
            return (
                m_scen.sound_power_level_identical_fans[i, j, types(i,j,1), f]
                - m_scen.sound_power_level_identical_fans[i, j, types(i,j,2), f]
            )
        return (
            m_scen.sound_power_level_in_level_addition[i, j, ind - 1, f]
            - m_scen.sound_power_level_identical_fans[i, j, types(i,j,ind + 1), f]
        )

    @m_scen.Expression(
        m_scen.level_add_set_multi,
        m_scen.polyhedral_coeff_set_multi,
        model.intervals,
        doc="",
    )
    def linear_diff1_multi(m_scen, i, j, ind, t, f):
        return (
            m_scen.polyhedral_approx_slope_multi[t]
            * (m_scen.difference_multi[i, j, ind, f])
            + m_scen.polyhedral_approx_y_intercept_multi[t]
        )

    @m_scen.Expression(
        m_scen.level_add_set_multi,
        m_scen.polyhedral_coeff_set_multi,
        model.intervals,
        doc="",
    )
    def linear_diff2_multi(m_scen, i, j, ind, t, f):
        return (
            m_scen.polyhedral_approx_slope_multi[t]
            * (-m_scen.difference_multi[i, j, ind, f])
            + m_scen.polyhedral_approx_y_intercept_multi[t]
        )

    # Constraint

    @m_scen.Constraint(
        m_scen.level_add_set_multi,
        m_scen.polyhedral_coeff_set_multi,
        model.intervals,
    )
    def level_difference1_multi(m_scen, i, j, ind, t, f):
        return (
            m_scen.level_increase1_multi[i, j, ind, f]
            >= m_scen.linear_diff1_multi[i, j, ind, t, f]
        )

    @m_scen.Constraint(
        m_scen.level_add_set_multi,
        m_scen.polyhedral_coeff_set_multi,
        model.intervals,
    )
    def level_difference2_multi(m_scen, i, j, ind, t, f):
        return (
            m_scen.level_increase2_multi[i, j, ind, f]
            >= m_scen.linear_diff2_multi[i, j, ind, t, f]
        )

    @m_scen.Constraint(m_scen.level_add_set_multi, model.intervals, doc="")
    def sound_power_level_out_with_linear_polyhedrals1_multi(m_scen, i, j, ind, f):
        if ind == 1:
            return (
                m_scen.sound_power_level_in_level_addition[i, j, 1, f]
                >= m_scen.sound_power_level_identical_fans[i, j, types(i,j,1), f]
                + m_scen.level_increase1_multi[i, j, 1, f]
            )
        return (
            m_scen.sound_power_level_in_level_addition[i, j, ind, f]
            >= m_scen.sound_power_level_in_level_addition[i, j, ind -1, f] + m_scen.level_increase1_multi[i, j, ind, f]
        )

    @m_scen.Constraint(m_scen.level_add_set_multi, model.intervals, doc="")
    def sound_power_level_out_with_linear_polyhedrals2_multi(m_scen, i, j, ind, f):
        if ind == 1:
            return (
                m_scen.sound_power_level_in_level_addition[i, j, 1, f]
                >= m_scen.sound_power_level_identical_fans[i, j, types(i,j,2), f]
                + m_scen.level_increase2_multi[i, j, 1, f]
            )
        return (
            m_scen.sound_power_level_in_level_addition[i, j, ind, f]
            >= m_scen.sound_power_level_identical_fans[i, j, types(i,j,ind + 1), f]
            + m_scen.level_increase2_multi[i, j, ind, f]
        )

    @m_scen.Constraint(model.E_fan_station_single, model.intervals, doc="")
    def sound_power_level_out_fan_station_singles(m_scen, i, j, f):
        return (
            m_scen.sound_power_level_in_level_addition[i, j, 1, f] == m_scen.sound_power_level_identical_fans[i, j, types(i,j,1), f]
        )

    return m_scen




# def level_add_identical_fans(block, expression, no_lvl_add_index, lvl_add_index):
#     """
#     Level adding over one of the indices of a multi-indexed expression while keeping the other index. The final value is the last block.w value.
#     """

#     types = lambda x: lvl_add_index.at(x)

#     n_expr = int(len(expression) / len(no_lvl_add_index))

#     block.level_add_set_multi = pyo.RangeSet(
#         n_expr - 1, doc="set used for tournament style level addition"
#     )

#     block.polyhedral_coeff_set_multi = pyo.RangeSet(
#         3, doc="set used for the coefficients of the polyhedral approximation"
#     )

#     block.polyhedral_approx_slope_multi = pyo.Param(
#         block.polyhedral_coeff_set_multi,
#         initialize={1: -0.415, 2: -0.219, 3: -0.066},
#         doc="slope of the linear polyhedral approximation of the level rise",
#     )

#     block.polyhedral_approx_y_intercept_multi = pyo.Param(
#         block.polyhedral_coeff_set_multi,
#         initialize={1: 2.943, 2: 2.288, 3: 1.056},
#         doc="y intercept of the linear polyhedral approximation of the level rise",
#     )

#     # Variables

#     block.level_increase1_multi = pyo.Var(
#         block.level_add_set_multi,
#         no_lvl_add_index,
#         within=pyo.NonNegativeReals,
#         bounds=(0, block.max_sound_power_level),
#     )

#     block.level_increase2_multi = pyo.Var(
#         block.level_add_set_multi,
#         no_lvl_add_index,
#         within=pyo.NonNegativeReals,
#         bounds=(0, block.max_sound_power_level),
#     )

#     block.w = pyo.Var(
#         block.level_add_set_multi,
#         no_lvl_add_index,
#         within=pyo.Reals,
#         bounds=(-20, block.max_sound_power_level),
#         doc="Helper variable used for tournament style level addition",
#     )

#     # Expression

#     @block.Expression(block.level_add_set_multi, no_lvl_add_index)
#     def difference_multi(block, i, ind):
#         if i == 1:
#             return expression[types(1), ind] - expression[types(2), ind]
#         return block.w[i - 1, ind] - expression[types(i + 1), ind]

#     @block.Expression(
#         block.polyhedral_coeff_set_multi,
#         block.level_add_set_multi,
#         no_lvl_add_index,
#         doc="",
#     )
#     def linear_diff1_multi(block, t, i, ind):
#         return (
#             block.polyhedral_approx_slope_multi[t] * (block.difference_multi[i, ind])
#             + block.polyhedral_approx_y_intercept_multi[t]
#         )

#     @block.Expression(
#         block.polyhedral_coeff_set_multi,
#         block.level_add_set_multi,
#         no_lvl_add_index,
#         doc="",
#     )
#     def linear_diff2_multi(block, t, i, ind):
#         return (
#             block.polyhedral_approx_slope_multi[t] * (-block.difference_multi[i, ind])
#             + block.polyhedral_approx_y_intercept_multi[t]
#         )

#     # Constraint

#     @block.Constraint(
#         block.polyhedral_coeff_set_multi, block.level_add_set_multi, no_lvl_add_index
#     )
#     def level_difference1_multi(block, t, i, ind):
#         return (
#             block.level_increase1_multi[i, ind] >= block.linear_diff1_multi[t, i, ind]
#         )

#     @block.Constraint(
#         block.polyhedral_coeff_set_multi, block.level_add_set_multi, no_lvl_add_index
#     )
#     def level_difference2_multi(block, t, i, ind):
#         return (
#             block.level_increase2_multi[i, ind] >= block.linear_diff2_multi[t, i, ind]
#         )

#     @block.Constraint(block.level_add_set_multi, no_lvl_add_index, doc="")
#     def sound_power_level_out_with_linear_polyhedrals1_multi(block, i, ind):
#         if i == 1:
#             return (
#                 block.w[1, ind]
#                 >= expression[types(1), ind] + block.level_increase1_multi[1, ind]
#             )
#         return (
#             block.w[i, ind] >= block.w[i - 1, ind] + block.level_increase1_multi[i, ind]
#         )

#     @block.Constraint(block.level_add_set_multi, no_lvl_add_index, doc="")
#     def sound_power_level_out_with_linear_polyhedrals2_multi(block, i, ind):
#         if i == 1:
#             return (
#                 block.w[1, ind]
#                 >= expression[types(2), ind] + block.level_increase2_multi[1, ind]
#             )
#         return (
#             block.w[i, ind]
#             >= expression[types(i + 1), ind] + block.level_increase2_multi[i, ind]
#         )

#     return block
