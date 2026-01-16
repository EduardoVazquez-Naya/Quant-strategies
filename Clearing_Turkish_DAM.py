# ==============================================================================
# ENERGY MARKET CLEARING ALGORITHM - ITERATIVE METHOD
# ==============================================================================
# This algorithm clears an electricity market using an iterative approach to
# handle complex bid types including hourly, block, and flexible bids.
# ==============================================================================
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ==============================================================================
# PERFORMANCE TRACKING
# ==============================================================================
start_time = time.time()

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
# Load market bid data from CSV file
# TODO: Update this path to match your file location
df = pd.read_csv(r"C:\Users\evzqn\OneDrive\Documents\MiF 2025-2026\Quant Reading\Python Projects\Veri_4.in", header=None)

# Extract columns from dataframe
N = df[0].tolist()    # Bid ID numbers
B = df[3].tolist()    # Bid type (S=Simple/Hourly, B=Block, F=Flexible)
S = df[2].tolist()    # Session/Hour (1-24)
Q = df[4].tolist()    # Quantity (positive=demand, negative=supply)
P = df[5].tolist()    # Price
L = df[6].tolist()    # Length/Duration (for block bids)
pnt = df[7].tolist()  # Parent bid ID (for linked bids)

# ==============================================================================
# BID CATEGORIZATION
# ==============================================================================
# Separate bids into different categories for processing
BR = []   # Block bids (rejected initially)
FR = []   # All flexible bids
FD = []   # Flexible demand bid IDs
FS = []   # Flexible supply bid IDs

for i in range(len(B)):
    if B[i] == 'B':
        # Block bids: [Price, Type, Session, Quantity, Length, ID, Parent]
        BR.append([P[i], B[i], S[i], Q[i], L[i], N[i], pnt[i]])
    
    if B[i] == 'F':
        # All flexible bids
        FR.append([P[i], B[i], S[i], Q[i], L[i], N[i]])
        
        # Separate flexible demand (Q>0) and supply (Q<0)
        if Q[i] > 0:
            FD.append(N[i])
        if Q[i] < 0:
            FS.append(N[i])

# ==============================================================================
# ITERATION PARAMETERS
# ==============================================================================
iteration_number = 0
iteration_limit = 20           # Max iterations for block bid acceptance
grand_iteration_number = 0
grand_iteration_limit = 10     # Max iterations for overall algorithm

# ==============================================================================
# PHASE 1: INITIAL CLEARING PRICES (HOURLY BIDS ONLY)
# ==============================================================================
# Calculate initial clearing price for each of 24 hours using only simple bids

C_P = []   # Clearing prices (one per hour)
AX = []    # Aggregated supply-demand curves for each hour

for j in range(1, 25):  # For each hour (1-24)
    XD = []  # Demand points [Price, Quantity]
    XS = []  # Supply points [Price, Quantity]
    X = []   # Combined points [Price]
    
    # Collect all simple bids for this hour
    for i in range(len(B)):
        if B[i] == 'S' and S[i] == j:
            X.append([P[i]])
            if Q[i] > 0:  # Demand
                XD.append([P[i], Q[i]])
            if Q[i] < 0:  # Supply
                XS.append([P[i], Q[i]])
    
    # Sort points by price
    X.sort()
    XD.sort()
    XS.sort()
    
    # Skip if no bids for this hour
    if len(X) < 2 or len(XD) == 0 or len(XS) == 0:
        C_P.append(0)
        AX.append(X)
        continue
    
    # Calculate net quantity at first two price points
    X[0].append(XD[0][1] + XS[0][1])
    X[1].append(XD[0][1] + XS[0][1])
    
    # Interpolate net quantity for intermediate price points
    for i in range(2, len(X) - 2):
        kd = 0  # Demand curve index
        ks = 0  # Supply curve index
        
        # FIXED: Find demand curve segment containing current price
        # Added bounds checking to prevent index errors
        while kd < len(XD) - 1 and XD[kd][0] <= X[i][0]:
            kd += 1
        
        # FIXED: Find supply curve segment containing current price
        # Added bounds checking to prevent index errors
        while ks < len(XS) - 1 and XS[ks][0] <= X[i][0]:
            ks += 1
        
        # FIXED: Use X[i][0] (scalar) instead of X[i] (list)
        # Interpolate demand and supply quantities at current price
        demand_qty = np.interp(X[i][0], 
                              [XD[kd-1][0], XD[kd][0]], 
                              [XD[kd-1][1], XD[kd][1]])
        supply_qty = np.interp(X[i][0], 
                              [XS[ks-1][0], XS[ks][0]], 
                              [XS[ks-1][1], XS[ks][1]])
        
        # Net quantity = demand + supply (supply is negative)
        X[i].append(float(demand_qty + supply_qty))
    
    # Set net quantity at last two price points
    X[-1].append(XD[-1][1] + XS[-1][1])
    X[-2].append(XD[-1][1] + XS[-1][1])
    
    # Store aggregated curve for this hour
    AX.append(X)
    
    # Binary search to find clearing price (where net quantity = 0)
    a = 0
    b = len(X) - 1
    while abs(b - a) > 1:
        c = int(a + (b - a) // 2)
        if X[c][1] >= 0:
            a = c
        else:
            b = c
    
    # FIXED: Handle case where clearing price might be outside bid range
    if a == b or abs(b - a) != 1:
        C_P.append((X[a][0] + X[b][0]) / 2)
    else:
        # Interpolate exact clearing price
        clearing_price = np.interp(0, 
                                   [X[b][1], X[a][1]], 
                                   [X[b][0], X[a][0]])
        C_P.append(clearing_price)

print("Initial Clearing Prices:", C_P)

# ==============================================================================
# PREPARE FLEXIBLE BIDS FOR PROCESSING
# ==============================================================================
FRS = []  # Flexible supply bids [Price, Quantity, ID, AssignedHour]
FRD = []  # Flexible demand bids [Price, Quantity, ID, AssignedHour]
FKS = []  # Accepted flexible supply bids
FKD = []  # Accepted flexible demand bids

for i in range(len(B)):
    if B[i] == 'F' and Q[i] < 0:
        FRS.append([P[i], Q[i], N[i], 0])
    if B[i] == 'F' and Q[i] > 0:
        FRD.append([P[i], Q[i], N[i], 0])

# ==============================================================================
# CALCULATE BLOCK BID INCREMENTAL VALUES
# ==============================================================================
# For each block bid, calculate profitability based on average clearing price
# over its duration

for i in range(len(BR)):
    C_P_avg = 0
    start_hour = BR[i][2] - 1
    duration = BR[i][4]
    
    # FIXED: Add bounds checking for hour range
    end_hour = min(start_hour + duration, 24)
    
    # Sum clearing prices over block duration
    for j in range(start_hour, end_hour):
        C_P_avg += C_P[j]
    
    # Calculate average price
    C_P_avg = C_P_avg / duration
    
    # Incremental Value = (Bid Price - Average Clearing Price) * |Quantity| * Duration
    incremental_value = (BR[i][0] - C_P_avg) * abs(BR[i][3]) * BR[i][4]
    BR[i].append(incremental_value)

# Reorder to put incremental value first: [IV, P, B, S, Q, L, N, pnt]
for i in range(len(BR)):
    IV = BR[i][7]
    del BR[i][7]
    BR[i].insert(0, IV)

# ==============================================================================
# INITIALIZE ACCEPTED/REJECTED BID LISTS
# ==============================================================================
BK = []    # Accepted block bids

# Hourly block bid adjustments (net quantity added by accepted block bids)
HBK = [0.0] * 24     # Total adjustment per hour
HBKN = [0.0] * 24    # Supply (negative) adjustment per hour
HBKP = [0.0] * 24    # Demand (positive) adjustment per hour

# Sort block bids by incremental value (highest first)
BR.sort(reverse=True)  # Format: [IV, P, B, S, Q, L, N, pnt]

# Accept the most profitable block bid initially (if any exist)
if len(BR) > 0:
    BK.append(BR[0])
    del BR[0]

# ==============================================================================
# HELPER FUNCTION: Calculate clearing price for a given hour
# ==============================================================================
def calculate_clearing_price(hour_idx, curve, adjustment):
    """
    Binary search to find clearing price where supply = demand
    
    Args:
        hour_idx: Hour index (0-23)
        curve: Price-quantity curve for this hour
        adjustment: Net quantity adjustment from accepted block/flex bids
    
    Returns:
        Clearing price
    """
    if len(curve) < 2:
        return 0
    
    a = 0
    b = len(curve) - 1
    
    # Binary search for zero crossing
    while abs(b - a) > 1:
        c = (a + b) // 2
        if curve[c][1] + adjustment >= 0:
            a = c
        else:
            b = c
    
    # Interpolate exact clearing price
    if abs(b - a) == 1:
        return float(np.interp(0, 
                              [curve[b][1] + adjustment, curve[a][1] + adjustment],
                              [curve[b][0], curve[a][0]]))
    else:
        return (curve[a][0] + curve[b][0]) / 2

# ==============================================================================
# PHASE 2: ITERATIVE BLOCK BID ACCEPTANCE
# ==============================================================================
print("\n--- Starting Block Bid Acceptance ---")

while grand_iteration_number < grand_iteration_limit:
    
    iteration_number = 0
    
    # ---------------------------------------------------------------------------
    # INNER LOOP: Accept profitable block bids
    # ---------------------------------------------------------------------------
    while iteration_number < iteration_limit:
        
        # Remove unprofitable accepted blocks (negative incremental value)
        i = 0
        while i < len(BK):
            if BK[i][0] < 0:
                BR.append(BK[i])
                del BK[i]
            else:
                i += 1
        
        # Re-sort rejected bids by profitability
        BR.sort(reverse=True)
        iteration_number += 1
        
        # Accept profitable block bids one by one
        while len(BK) > 0 and len(BR) > 0 and BK[0][0] > 0:
            
            # Update hourly adjustments for newly accepted block
            start_hour = BK[0][3] - 1
            duration = BK[0][5]
            quantity = BK[0][4]
            
            # FIXED: Add bounds checking
            end_hour = min(start_hour + duration, 24)
            
            for j in range(start_hour, end_hour):
                HBK[j] += quantity
                if quantity > 0:
                    HBKP[j] += quantity
                else:
                    HBKN[j] += quantity
            
            # Recalculate clearing prices with updated hourly adjustments
            for j in range(24):
                C_P[j] = calculate_clearing_price(j, AX[j], HBK[j])
            
            # Recalculate incremental values for all rejected block bids
            for i in range(len(BR)):
                C_P_avg = 0
                start_hour = BR[i][3] - 1
                duration = BR[i][5]
                end_hour = min(start_hour + duration, 24)
                
                for j in range(start_hour, end_hour):
                    C_P_avg += C_P[j]
                
                C_P_avg = C_P_avg / duration
                BR[i][0] = (BR[i][1] - C_P_avg) * abs(BR[i][4]) * BR[i][5]
            
            # Re-sort and accept next most profitable block
            BR.sort(reverse=True)
            if len(BR) > 0:
                BK.append(BR[0])
                del BR[0]
                BK.reverse()
    
    print(f"Iteration {grand_iteration_number + 1}: {len(BK)} blocks accepted")
    
    # ---------------------------------------------------------------------------
    # HANDLE LINKED BIDS
    # ---------------------------------------------------------------------------
    # If a parent block is rejected, also reject its children
    
    changes_made = True
    while changes_made:
        changes_made = False
        
        for i in range(len(BK)):
            for j in range(len(BR)):
                # Check if rejected bid is parent of accepted bid
                if BR[j][6] == BK[i][7]:
                    print(f"Rejecting linked bid: {BK[i][6]}")
                    
                    # Reject the child bid
                    BR.append(BK[i])
                    
                    # Remove child's effect from hourly adjustments
                    start_hour = BK[i][3] - 1
                    duration = BK[i][5]
                    quantity = BK[i][4]
                    end_hour = min(start_hour + duration, 24)
                    
                    for k in range(start_hour, end_hour):
                        HBK[k] -= quantity
                        if quantity > 0:
                            HBKP[k] -= quantity
                        else:
                            HBKN[k] -= quantity
                    
                    del BK[i]
                    changes_made = True
                    
                    # Recalculate clearing prices
                    for z in range(24):
                        C_P[z] = calculate_clearing_price(z, AX[z], HBK[z])
                    
                    break
            if changes_made:
                break
    
    # ---------------------------------------------------------------------------
    # PHASE 3: FLEXIBLE BID ASSIGNMENT
    # ---------------------------------------------------------------------------
    # Assign flexible bids to hours with most favorable prices
    
    FRS.sort()  # Sort flexible supply by price (lowest first)
    FRD.sort(reverse=True)  # Sort flexible demand by price (highest first)
    
    # Assign flexible demand bids to lowest-price hour
    while len(FRD) > 0:
        # Find hour with minimum clearing price
        C_Pmin = min(C_P)
        Smin = C_P.index(C_Pmin) + 1
        
        # Check if this flexible demand bid is profitable
        if FRD[0][0] <= C_Pmin:
            break
        
        # Accept and assign to minimum price hour
        FKD.append(FRD[0])
        FKD[-1][3] = Smin
        
        # Update hourly adjustments
        quantity = FRD[0][1]
        HBK[Smin - 1] += quantity
        if quantity > 0:
            HBKP[Smin - 1] += quantity
        else:
            HBKN[Smin - 1] += quantity
        
        del FRD[0]
        
        # Recalculate clearing price for the affected hour
        C_P[Smin - 1] = calculate_clearing_price(Smin - 1, AX[Smin - 1], HBK[Smin - 1])
    
    # Assign flexible supply bids to highest-price hour
    while len(FRS) > 0:
        # Find hour with maximum clearing price
        C_Pmax = max(C_P)
        Smax = C_P.index(C_Pmax) + 1
        
        # Check if this flexible supply bid is profitable
        if FRS[0][0] >= C_Pmax:
            break
        
        # Accept and assign to maximum price hour
        FKS.append(FRS[0])
        FKS[-1][3] = Smax
        
        # Update hourly adjustments
        quantity = FRS[0][1]
        HBK[Smax - 1] += quantity
        if quantity > 0:
            HBKP[Smax - 1] += quantity
        else:
            HBKN[Smax - 1] += quantity
        
        del FRS[0]
        
        # Recalculate clearing price for the affected hour
        C_P[Smax - 1] = calculate_clearing_price(Smax - 1, AX[Smax - 1], HBK[Smax - 1])
    
    grand_iteration_number += 1

# ==============================================================================
# PHASE 4: FEASIBILITY CHECK
# ==============================================================================
# Accept any remaining profitable non-linked block bids

print("\n--- Final Feasibility Check ---")

i = 0
while i < len(BR):
    if BR[i][0] > 0:
        # Accept non-linked profitable blocks
        if str(BR[i][7]) == 'nan' or str(BR[i][7]) == 'NaN':
            print(f"Accepting profitable non-linked block: {BR[i][6]}")
            BK.append(BR[i])
            
            # Update hourly adjustments
            start_hour = BR[i][3] - 1
            duration = BR[i][5]
            quantity = BR[i][4]
            end_hour = min(start_hour + duration, 24)
            
            for j in range(start_hour, end_hour):
                HBK[j] += quantity
                if quantity > 0:
                    HBKP[j] += quantity
                else:
                    HBKN[j] += quantity
            
            del BR[i]
            
            # Recalculate clearing prices
            for j in range(24):
                C_P[j] = calculate_clearing_price(j, AX[j], HBK[j])
            
            # Recalculate incremental values for remaining blocks
            for k in range(len(BR)):
                C_P_avg = 0
                start_hour = BR[k][3] - 1
                duration = BR[k][5]
                end_hour = min(start_hour + duration, 24)
                
                for j in range(start_hour, end_hour):
                    C_P_avg += C_P[j]
                
                C_P_avg = C_P_avg / duration
                BR[k][0] = (BR[k][1] - C_P_avg) * abs(BR[k][4]) * BR[k][5]
        else:
            i += 1
    else:
        i += 1

# ==============================================================================
# OUTPUT RESULTS
# ==============================================================================
print("\n" + "="*80)
print("FINAL MARKET CLEARING RESULTS")
print("="*80)
print(f"\nFinal Clearing Prices (€/MWh): {[round(cp, 2) for cp in C_P]}")

end_time = time.time() - start_time
print(f"\nComputation Time: {end_time:.2f} seconds")

print(f"\nAccepted Flexible Demand Bids: {len(FKD)}")
print(f"Accepted Flexible Supply Bids: {len(FKS)}")
print(f"Accepted Block Bids: {len(BK)}")

# ==============================================================================
# GENERATE SOLUTION VARIABLES FOR OPTIMIZATION MODEL
# ==============================================================================

# Track accepted bids
acc = []

# Accepted block bid IDs
for i in range(len(BK)):
    acc.append(BK[i][6])

# Accepted flexible demand bids [ID, AssignedHour]
for i in range(len(FKD)):
    acc.append([FKD[i][2], FKD[i][3]])

# Accepted flexible supply bids [ID, AssignedHour]
for i in range(len(FKS)):
    acc.append([FKS[i][2], FKS[i][3]])

# Count different bid types
hourlen = sum(1 for b in B if b == 'S')
blocklen = sum(1 for b in B if b == 'B')
flexlenD = sum(1 for i, b in enumerate(B) if b == 'F' and Q[i] > 0)
flexlenS = sum(1 for i, b in enumerate(B) if b == 'F' and Q[i] < 0)
flexlen = flexlenD + flexlenS

# Create binary solution variables for block bids
solvarBD = []  # Block demand acceptance (1=accepted, 0=rejected)
solvarBS = []  # Block supply acceptance

for i in range(hourlen, hourlen + blocklen):
    if N[i] in acc and Q[i] > 0:
        solvarBD.append(1)
    elif Q[i] > 0:
        solvarBD.append(0)
    
    if N[i] in acc and Q[i] < 0:
        solvarBS.append(1)
    elif Q[i] < 0:
        solvarBS.append(0)

# Process flexible bid assignments
FaccD = [fkd[2] for fkd in FKD]  # Accepted flexible demand IDs
FaccDH = [fkd[3] for fkd in FKD]  # Their assigned hours

FaccS = [fks[2] for fks in FKS]  # Accepted flexible supply IDs
FaccSH = [fks[3] for fks in FKS]  # Their assigned hours

# Create hour-by-bid acceptance matrix for flexible bids
AFacc = [[[0] * flexlenD, [0] * flexlenS] for _ in range(24)]

# Populate flexible demand acceptances
for i in range(len(FaccD)):
    if FaccD[i] in FD:
        hour_idx = FaccDH[i] - 1
        flex_idx = FD.index(FaccD[i])
        AFacc[hour_idx][0][flex_idx] = 1

# Populate flexible supply acceptances
for i in range(len(FaccS)):
    if FaccS[i] in FS:
        hour_idx = FaccSH[i] - 1
        flex_idx = FS.index(FaccS[i])
        AFacc[hour_idx][1][flex_idx] = 1

print(f"\nBlock Demand Acceptances: {sum(solvarBD)}/{len(solvarBD)}")
print(f"Block Supply Acceptances: {sum(solvarBS)}/{len(solvarBS)}")

# Restructure flex acceptance matrix
NAFacc = [[], []]
for i in range(24):
    NAFacc[0].append(AFacc[i][0])
    NAFacc[1].append(AFacc[i][1])

# ==============================================================================
# CALCULATE HOURLY BID ACCEPTANCES & WELFARE
# ==============================================================================
HS = [[] for _ in range(24)]  # Hourly supply acceptances
HD = [[] for _ in range(24)]  # Hourly demand acceptances

for j in range(24):
    for i in range(hourlen):
        # Demand bids: accepted if price >= clearing price
        if S[i] == j + 1 and Q[i] > 0:
            if P[i] >= C_P[j]:
                HD[j].append([1, P[i]])
            else:
                HD[j].append([0, P[i]])
        
        # Supply bids: accepted if price <= clearing price
        if S[i] == j + 1 and Q[i] < 0:
            if P[i] <= C_P[j]:
                HS[j].append([1, P[i]])
            else:
                HS[j].append([0, P[i]])

# Handle partial acceptances at marginal price
for j in range(24):
    # Demand bids
    for i in range(len(HD[j]) - 1):
        if HD[j][i][0] == 0 and HD[j][i+1][0] == 1:
            # Partially accepted bid
            partial_acceptance = (HD[j][i][1] - C_P[j]) / (HD[j][i][1] - HD[j][i+1][1])
            HD[j][i+1][0] = partial_acceptance
    
    # Supply bids
    for i in range(len(HS[j]) - 1):
        if HS[j][i][0] == 1 and HS[j][i+1][0] == 0:
            # Partially accepted bid
            partial_acceptance = (C_P[j] - HS[j][i][1]) / (HS[j][i+1][1] - HS[j][i][1])
            HS[j][i][0] = partial_acceptance

# Extract acceptance values only
for j in range(24):
    HD[j] = [hd[0] for hd in HD[j]]
    HS[j] = [hs[0] for hs in HS[j]]
    
    # Remove first element (seems to be legacy code)
    if len(HD[j]) > 0:
        HD[j].pop(0)
    if len(HS[j]) > 0:
        HS[j].pop(0)

# ==============================================================================
# CALCULATE SOCIAL WELFARE (SIMPLIFIED - USING INTERNAL DATA)
# ==============================================================================
print("\n" + "="*80)
print("SOCIAL WELFARE CALCULATION")
print("="*80)

# Build bid dataframes from the data we already loaded
# This replaces the external function calls that were reading from Veri_10.in

# Hourly Supply Bids
hourly_supply_bids = []
for i in range(len(B)):
    if B[i] == 'S' and Q[i] < 0:
        hourly_supply_bids.append({
            'hour': S[i],
            'price': P[i],
            'quantity': abs(Q[i]),
            'id': N[i]
        })

# Hourly Demand Bids
hourly_demand_bids = []
for i in range(len(B)):
    if B[i] == 'S' and Q[i] > 0:
        hourly_demand_bids.append({
            'hour': S[i],
            'price': P[i],
            'quantity': Q[i],
            'id': N[i]
        })

# Block Supply Bids (from solvarBS)
block_supply_welfare = 0
block_idx = 0
for i in range(len(B)):
    if B[i] == 'B' and Q[i] < 0:
        if block_idx < len(solvarBS) and solvarBS[block_idx] == 1:
            # Accepted block supply bid
            block_supply_welfare += abs(Q[i]) * P[i] * L[i]
        block_idx += 1

# Block Demand Bids (from solvarBD)
block_demand_welfare = 0
block_idx = 0
for i in range(len(B)):
    if B[i] == 'B' and Q[i] > 0:
        if block_idx < len(solvarBD) and solvarBD[block_idx] == 1:
            # Accepted block demand bid
            block_demand_welfare += Q[i] * P[i] * L[i]
        block_idx += 1

print(f"Block Demand Welfare: {block_demand_welfare:.2f}")
print(f"Block Supply Welfare: {block_supply_welfare:.2f}")

# Flexible bid welfare
flex_demand_welfare = 0
for i in range(len(FKD)):
    flex_demand_welfare += abs(FKD[i][1]) * FKD[i][0]

flex_supply_welfare = 0
for i in range(len(FKS)):
    flex_supply_welfare += abs(FKS[i][1]) * FKS[i][0]

print(f"Flexible Demand Welfare: {flex_demand_welfare:.2f}")
print(f"Flexible Supply Welfare: {flex_supply_welfare:.2f}")

# Calculate hourly bid welfare
hourly_supply_welfare = 0
for j in range(24):
    hour_bids = [b for b in hourly_supply_bids if b['hour'] == j + 1]
    hour_bids.sort(key=lambda x: x['price'])
    
    for bid in hour_bids:
        if bid['price'] <= C_P[j]:
            # Accepted supply bid
            hourly_supply_welfare += bid['quantity'] * bid['price']

hourly_demand_welfare = 0
for j in range(24):
    hour_bids = [b for b in hourly_demand_bids if b['hour'] == j + 1]
    hour_bids.sort(key=lambda x: x['price'], reverse=True)
    
    for bid in hour_bids:
        if bid['price'] >= C_P[j]:
            # Accepted demand bid
            hourly_demand_welfare += bid['quantity'] * bid['price']

print(f"Hourly Supply Welfare: {hourly_supply_welfare:.2f}")
print(f"Hourly Demand Welfare: {hourly_demand_welfare:.2f}")

# Calculate total social welfare
# Welfare = Consumer Surplus + Producer Surplus
# = (Demand welfare - payments) + (payments - Supply costs)
# = Demand welfare - Supply costs
total_welfare = (hourly_demand_welfare + block_demand_welfare + flex_demand_welfare - 
                 hourly_supply_welfare - block_supply_welfare - flex_supply_welfare)

print("\n" + "="*80)
print(f"TOTAL SOCIAL WELFARE: {total_welfare:.2f}")
print("="*80)

# ==============================================================================
# EXPORT RESULTS (Optional)
# ==============================================================================
# Uncomment to save results to CSV files

# # Save clearing prices
# pd.DataFrame({
#     'Hour': range(1, 25),
#     'Clearing_Price': C_P
# }).to_csv('clearing_prices.csv', index=False)

# # Save accepted block bids
# pd.DataFrame(BK, columns=['IV', 'Price', 'Type', 'Start_Hour', 'Quantity', 
#                           'Duration', 'ID', 'Parent']).to_csv('accepted_blocks.csv', index=False)

# # Save accepted flexible bids
# pd.DataFrame(FKD, columns=['Price', 'Quantity', 'ID', 'Assigned_Hour']).to_csv('accepted_flex_demand.csv', index=False)
# pd.DataFrame(FKS, columns=['Price', 'Quantity', 'ID', 'Assigned_Hour']).to_csv('accepted_flex_supply.csv', index=False)

print("\n" + "="*80)
print("ALGORITHM COMPLETE")
print("="*80)