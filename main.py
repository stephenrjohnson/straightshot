import argparse
import concurrent.futures
import itertools
import logging
import os

import matplotlib.pyplot as plt
import osmnx as ox

logging.basicConfig(level=logging.INFO)
import multiprocessing as mp

from gpxpy.gpx import GPX, GPXTrack, GPXTrackPoint, GPXTrackSegment
from shapely.geometry import MultiPolygon, Point


def display(routes, G, gdf, edgenodes, displyedgenodes, place):
    logging.info("Sorted routes by straighntess number: %s", len(routes))
    fig, ax = ox.plot_graph_routes(
        G,
        routes,
        route_colors=["r", "y", "g"],
        route_linewidth=1,
        show=False,
        close=False,
    )
    fig.canvas.manager.set_window_title(place)
    gdf.plot(ax=ax, fc="k", ec="#666666", lw=1, alpha=1, zorder=-1)
    if displyedgenodes:
        subG = G.subgraph(edgenodes)
        logging.info("Plotting closest nodes to the edge")
        ox.plot_graph(subG, node_color="orange", show=False, ax=ax)

    margin = 0.02
    west, south, east, north = gdf.unary_union.bounds
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))
    plt.show()


def is_route_straight(route, G, tolerance=9):
    first_point = route[0]
    last_point = route[-1]
    route_bearing = ox.bearing.calculate_bearing(
        G.nodes[first_point]["y"],
        G.nodes[first_point]["x"],
        G.nodes[last_point]["y"],
        G.nodes[last_point]["x"],
    )
    for i in range(1, len(route) - 1):
        point = route[i]
        bearing = ox.bearing.calculate_bearing(
            G.nodes[first_point]["y"],
            G.nodes[first_point]["x"],
            G.nodes[point]["y"],
            G.nodes[point]["x"],
        )
        if abs(bearing - route_bearing) > tolerance:
            return False
    return True


def main():
    args = parse_args()
    place = args.location
    logging.info("Generating strightline for %s", place)
    ox.config(log_console=True)
    gdf = ox.geocode_to_gdf(place)
    G = ox.graph_from_place(place, network_type=args.type, retain_all=False)

    closest_to_edge = work_out_edges(G, gdf)

    routes = []
    calculation = list(itertools.combinations(set(closest_to_edge), 2))
    threads = mp.cpu_count() - 1
    logging.info("Total Calculations %s on %s cores", len(calculation), threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count() - 1) as e:
        fut = [
            e.submit(ox.shortest_path, G, x[0], x[1], weight="length")
            for x in calculation
        ]
        for i, r in enumerate(concurrent.futures.as_completed(fut)):
            logging.info(
                "Calculated the shortest path: %s of total calculation: %s",
                i,
                len(calculation),
            )
            if is_route_straight(route=r.result(), G=G):
                routes.append(r.result())

    routes.sort(
        key=lambda route: sum(
            ox.utils_graph.get_route_edge_attributes(G, route, attribute="length")
        ),
        reverse=True,
    )
    display(routes[:3], G, gdf, closest_to_edge, args.display_edges, place)
    write_gpx(routes[:3], place, G)


def parse_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-l", "--location", help="Location to straight line")
    argParser.add_argument(
        "-e",
        "--display-edges",
        help="Display the edge nodes we have found",
        action="store_true",
    )
    argParser.add_argument(
        "-t",
        "--type",
        choices=["all", "bike", "drive", "drive_service", "bike"],
        default="walk",
        help="What type of route, bike, drive, walk, all, drive_service",
    )
    return argParser.parse_args()


def work_out_edges(G, gdf, edgebuffer=0.002):
    if type(gdf.geometry[0]) == MultiPolygon:
        exteriror = gdf.geometry[0].geoms[0]
    else:
        exteriror = gdf.geometry[0]

    # Find the closest nodes to the boundary of the place
    # closest_to_edge = set(ox.nearest_nodes(G, *exteriror.exterior.coords.xy))
    buffer = exteriror.exterior.buffer(edgebuffer)
    closest_to_edge = set(
        node
        for node, data in G.nodes(data=True)
        if buffer.intersects(Point(data["x"], data["y"]))
    )
    logging.info(
        "Found the closest points to the edge count is %s", len(closest_to_edge)
    )
    if len(closest_to_edge) > 5:
        return closest_to_edge

    logging.info("Doubling buffer to find more at end")
    return work_out_edges(G, gdf, edgebuffer=edgebuffer * 2)


def write_gpx(routes, place, G):
    if not os.path.exists("gpx"):
        os.makedirs("gpx")
    for i, route in enumerate(routes):
        gpx = GPX()
        gpx_track = GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_segment = GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        gpx_segment.points.extend(
            [GPXTrackPoint(G.nodes[point]["y"], G.nodes[point]["x"]) for point in route]
        )
        with open(f"gpx/{place.strip().replace(' ','_')}_{i}_route.gpx", "w") as f:
            f.write(gpx.to_xml())


if __name__ == "__main__":
    main()
