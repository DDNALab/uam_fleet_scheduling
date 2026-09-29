"""Independent aircraft-indexed benchmark for validating the hybrid projected MILP.

Use on SMALL identical model_input instances only. Passenger flows, failures, and capacity-adjustment counts are integer.
Booking-protection variables are continuous but admit an equivalent
integral allocation.

The reference uses passenger variables indexed by (aircraft, request period,
destination, departure period) instead of the paper's equivalent eligibility-Y
and departure-time-F encoding. Indexing only admissible assignments avoids big M
but represents the same detailed aircraft-level feasible operations.
"""
import math

import pulp

from .stochastic_model import charging_periods_required, _energy_needed_kwh, energy_price_per_kwh, _window


def build_detailed_model(model_input):
    p = model_input['parameters']
    scenarios = model_input['scenarios']
    bookings = model_input.get('advance_bookings', {})
    T = tuple(range(1, p['periods']+1))
    D = sorted({d for _,d in bookings} |
               {d for s in scenarios.values() for _,d in s['on_demand_demand']})
    cap = int(p.get('initial_fleet_max', 0))
    if cap > 6:
        raise ValueError('Aircraft-indexed reference is intentionally limited to initial_fleet_max <= 6')
    model = pulp.LpProblem('Detailed_Independent_UAM_Reference', pulp.LpMaximize)
    x0 = pulp.LpVariable('RefInitialFleet', lowBound=0, upBound=cap, cat='Integer')
    n = {(d,t):pulp.LpVariable(f'RefPlan_{di}_{t}',lowBound=0,cat='Integer')
         for di,d in enumerate(D) for t in T}
    b = {t:pulp.LpVariable(f'RefReserve_{t}',lowBound=0,cat='Integer') for t in T}
    q = {(r,d,k):pulp.LpVariable(f'RefProtected_r{r}_d{D.index(d)}_k{k}',lowBound=0,cat='Continuous')
         for r in T for d in D for k in _window(r,p['periods'],p['los_periods'])}

    for t in T:
        model += b[t] <= p['charging_facilities'], f'RefReserveLimit_{t}'
        model += pulp.lpSum(n[d,t] for d in D) <= p['max_departures'], f'RefPlanCap_{t}'
    for r in T:
        for d in D:
            model += (pulp.lpSum(q[r,d,k] for k in _window(r,p['periods'],p['los_periods']))
                      == bookings.get((r,d),0)), f'RefBooking_{r}_{D.index(d)}'
    for d in D:
        for k in T:
            model += pulp.lpSum(q[r,d,k] for r in T if k in _window(r,p['periods'],p['los_periods'])) <= p['evtol_capacity']*n[d,k], f'RefBookCap_{D.index(d)}_{k}'

    indexed = {}
    profits = {}
    losses = {}
    zeta = pulp.LpVariable('RefCVaR_zeta',lowBound=0)
    rho = {sid:pulp.LpVariable(f'RefCVaR_rho_s{sid}',lowBound=0) for sid in scenarios}
    for sid,s in scenarios.items():
        initial = [('initial',i) for i in range(cap)]
        incoming = [('incoming',j) for j in s['aircraft']]
        all_a = initial+incoming
        feasible = {}
        durations = {}
        for tag,j in incoming:
            info = s['aircraft'][j]
            durations[j] = charging_periods_required(info['initial_soc'],p)
            feasible[j] = [t for t in T if info['arrival_period'] <= t and t+durations[j] <= p['periods']]
        xs = {(j,t):pulp.LpVariable(f'RefCharge_s{sid}_j{j}_t{t}',lowBound=0,upBound=1,cat='Binary')
              for _,j in incoming for t in feasible[j]}
        zs = {(a,d,k):pulp.LpVariable(f'RefFlight_s{sid}_{a[0]}{a[1]}_d{D.index(d)}_k{k}',lowBound=0,upBound=1,cat='Binary')
              for a in all_a for d in D for k in T}
        plus = {(d,k):pulp.LpVariable(f'RefAdded_s{sid}_d{D.index(d)}_k{k}',lowBound=0,cat='Integer') for d in D for k in T}
        minus = {(d,k):pulp.LpVariable(f'RefCancelled_s{sid}_d{D.index(d)}_k{k}',lowBound=0,cat='Integer') for d in D for k in T}
        em = {t:pulp.LpVariable(f'RefEmergency_s{sid}_t{t}',lowBound=0,cat='Integer') for t in T}
        uns_b = {(r,d):pulp.LpVariable(f'RefUnservedBooked_s{sid}_r{r}_d{D.index(d)}',lowBound=0,cat='Integer')
                 for r in T for d in D}
        uns_o = {(r,d):pulp.LpVariable(f'RefUnservedOn_s{sid}_r{r}_d{D.index(d)}',lowBound=0,cat='Integer')
                 for r in T for d in D}
        yb = {(a,r,d,k):pulp.LpVariable(f'RefB_s{sid}_{a[0]}{a[1]}_r{r}_d{D.index(d)}_k{k}',lowBound=0,cat='Integer')
              for a in all_a for r in T for d in D for k in _window(r,p['periods'],p['los_periods'])}
        yo = {(a,r,d,k):pulp.LpVariable(f'RefO_s{sid}_{a[0]}{a[1]}_r{r}_d{D.index(d)}_k{k}',lowBound=0,cat='Integer')
              for a in all_a for r in T for d in D for k in _window(r,p['periods'],p['los_periods'])}
        # Initial fleet: at most the number pre-allocated in the first stage.
        model += pulp.lpSum(zs[a,d,k] for a in initial for d in D for k in T) <= x0, f'RefInitialDeployment_s{sid}'
        for a in initial:
            model += pulp.lpSum(zs[a,d,k] for d in D for k in T) <= 1, f'RefInitialOnce_s{sid}_{a[1]}'
        # Incoming aircraft: charge exactly once iff deployed, never depart before completion.
        for a in incoming:
            j=a[1]
            charge_count = pulp.lpSum(xs[j,t] for t in feasible[j])
            model += charge_count <= 1, f'RefAtMostOneCharge_s{sid}_j{j}'
            model += pulp.lpSum(zs[a,d,k] for d in D for k in T) == charge_count, f'RefChargeOnce_s{sid}_j{j}'
            for k in T:
                model += pulp.lpSum(zs[a,d,k] for d in D) <= pulp.lpSum(xs[j,t] for t in feasible[j] if t+durations[j] <= k), f'RefReady_s{sid}_j{j}_k{k}'
        for t in T:
            occupying = [xs[j,start] for _,j in incoming for start in feasible[j]
                         if durations[j]>0 and start<=t<start+durations[j]]
            model += pulp.lpSum(occupying) <= b[t]+em[t], f'RefChargingCap_s{sid}_t{t}'
            model += em[t] <= p['charging_facilities']-b[t], f'RefEmergencyCap_s{sid}_t{t}'
            model += pulp.lpSum(zs[a,d,t] for a in all_a for d in D) <= p['max_departures'], f'RefTakeoff_s{sid}_t{t}'
        for d in D:
            for k in T:
                model += pulp.lpSum(zs[a,d,k] for a in all_a) == n[d,k]+plus[d,k]-minus[d,k], f'RefReconcile_s{sid}_d{D.index(d)}_k{k}'
                model += minus[d,k] <= n[d,k], f'RefCancelBound_s{sid}_d{D.index(d)}_k{k}'
                # Every aircraft serves only its own destination/departure;
                # collectively, multiple request periods share its four seats.
                for a in all_a:
                    eligible_r=[r for r in T if k in _window(r,p['periods'],p['los_periods'])]
                    model += pulp.lpSum(yb[a,r,d,k]+yo[a,r,d,k] for r in eligible_r) <= p['evtol_capacity']*zs[a,d,k], f'RefAircraftSeats_s{sid}_{a[0]}{a[1]}_d{D.index(d)}_k{k}'
        for r in T:
            for d in D:
                window = _window(r,p['periods'],p['los_periods'])
                model += (pulp.lpSum(yb[a,r,d,k] for a in all_a for k in window) +uns_b[r,d] == bookings.get((r,d),0)), f'RefB_Balance_s{sid}_r{r}_d{D.index(d)}'
                model += (pulp.lpSum(yo[a,r,d,k] for a in all_a for k in window) +uns_o[r,d] == s['on_demand_demand'].get((r,d),0)), f'RefO_Balance_s{sid}_r{r}_d{D.index(d)}'
        revenue = pulp.lpSum(p['destination_fares'].get(d,0)*(yb[a,r,d,k]+yo[a,r,d,k])
                            for a in all_a for r in T for d in D for k in _window(r,p['periods'],p['los_periods']))
        chargecost=pulp.lpSum(xs[j,t]*_energy_needed_kwh(s['aircraft'][j]['initial_soc'],p)*energy_price_per_kwh(
                   (p['horizon']['start']+(t-1)*p['bin_size'])/60.,p) for _,j in incoming for t in feasible[j])
        costs=p['costs']
        flightcost=pulp.lpSum(costs['flight']*zs[a,d,k] for a in all_a for d in D for k in T)
        emcost=pulp.lpSum(costs['emergency_capacity']*em[t] for t in T)
        addcost=pulp.lpSum(costs['added_departure']*plus[d,k] for d in D for k in T)
        cancost=pulp.lpSum(costs['cancelled_departure']*minus[d,k] for d in D for k in T)
        bcost=pulp.lpSum(costs['unserved_booked']*uns_b[r,d] for r in T for d in D)
        ocost=pulp.lpSum(costs['unserved_on_demand']*uns_o[r,d] for r in T for d in D)
        profits[sid]=revenue-chargecost-flightcost-emcost-addcost-cancost-bcost-ocost
        losses[sid]=p['cvar']['booked_weight']*pulp.lpSum(uns_b.values())+p['cvar']['on_demand_weight']*pulp.lpSum(uns_o.values())
        model += rho[sid] >= losses[sid]-zeta, f'RefCVaRExcess_s{sid}'
        indexed[sid]={'initial':initial,'incoming':incoming,'charge':xs,'feasible':feasible,
                      'durations':durations,'z':zs,'yb':yb,'yo':yo,'unserved_booked':uns_b,
                      'unserved_on_demand':uns_o,'emergency':em,'added':plus,'cancelled':minus}
    expected=pulp.lpSum(s['probability']*profits[sid] for sid,s in scenarios.items())
    fleetcost=p['costs'].get('initial_fleet',0.0)*x0
    commitment=p['costs']['commitment']*pulp.lpSum(n.values())
    reservation=p['costs']['charging_reservation']*pulp.lpSum(b.values())
    cvar=zeta+(1./(1.-p['cvar']['alpha']))*pulp.lpSum(s['probability']*rho[sid] for sid,s in scenarios.items())
    model += expected-fleetcost-commitment-reservation-p['cvar']['weight']*cvar
    return {'model':model,'periods':T,'destinations':D,'scenarios':scenarios,'advance_bookings':bookings,
            'parameters':p,'variables':{'initial_fleet':x0,'n':n,'b':b,'protected_bookings':q,'scenarios':indexed,'zeta':zeta,'rho':rho},
            'expressions':{'scenario_profit':profits,'scenario_loss':losses,'expected_profit':expected,
                           'initial_fleet_cost':fleetcost,'commitment_cost':commitment,'reservation_cost':reservation,'cvar':cvar}}
