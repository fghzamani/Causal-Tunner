#!/usr/bin/env python3

import csv

def generate_parameter_combinations():
    global_planners = ["NavFn", "SmacHybrid", "SmacLattice", "Thetastar"]
    controllers = ["DWB", "MPPI"]
    footprint_types = [0, 1]
    footprint_0_values = [0.275, 0.3, 0.4, 0.5]
    # As a string, since CSV needs a textual representation
    footprint_1_value = "[[0.3,0.3], [-0.3,0.3], [-0.3,-1.3], [0.3, -1.3]]"
    inflation_radii = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    inflation_cost_scales = [0.5, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Create combinations
    for gp in global_planners:
        for ctrl in controllers:
            for ftype in footprint_types:
                if ftype == 0:
                    # For footprint_type=0, use each numerical footprint
                    for fp in footprint_0_values:
                        for ir in inflation_radii:
                            for ics in inflation_cost_scales:
                                yield [gp, ctrl, ftype, fp, ir, ics]
                else:
                    # For footprint_type=1, the footprint is the polygon
                    fp = footprint_1_value
                    for ir in inflation_radii:
                        for ics in inflation_cost_scales:
                            yield [gp, ctrl, ftype, fp, ir, ics]

def main():
    filename = "parameters.csv"
    header = ["Global Planner", "Controller", "Footprint_type", "Footprint", "Inflation_radius", "Inflation_cost_scale"]
    
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in generate_parameter_combinations():
            writer.writerow(row)

    print(f"CSV file '{filename}' generated successfully.")

if __name__ == "__main__":
    main()
