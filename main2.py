import argparse
import concurrent.futures
import itertools
import logging
import os
import multiprocessing as mp

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from gpxpy.gpx import GPX, GPXTrack, GPXTrackPoint, GPXTrackSegment
from shapely.geometry import MultiPolygon, Point, Polygon

logging.basicConfig(level=logging.INFO)


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


def shortest_staightest(G, x, weight="length", tolerance=9):
    route = ox.shortest_path(G, x[0], x[1], weight)
    if is_route_straight(route, G, tolerance):
        return route
    else:
        return None


def is_route_straight(route, G, tolerance):
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


def cache_graph(G, filename):
    if not os.path.exists("data"):
        os.makedirs("data")
    logging.info("Caching graph to %s", filename)
    ox.save_graphml(G, filename)


def load_cached_graph(filename):
    if os.path.isfile(filename):
        logging.info("Loading Cached graph from %s", filename)
        return ox.load_graphml(filename)
    else:
        return None


def calulate_routes(closest_to_edge, G):
    routes = []
    calculation = list(itertools.combinations(set(closest_to_edge), 2))
    threads = mp.cpu_count() - 1
    logging.info("Total Calculations %s on %s cores", len(calculation), threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as e:
        fut = [e.submit(shortest_staightest, G, x) for x in calculation]
        for i, r in enumerate(concurrent.futures.as_completed(fut)):
            logging.info(
                "Calculated the shortest path: %s of total calculation: %s",
                i,
                len(calculation),
            )
            if r.result():
                routes.append(r.result())
    return routes


def straightness_ratio(G, route):
    """
    Calculates the straightness ratio of a route.

    Args:
        G: The networkx graph.
        route: A list of node IDs representing the route.

    Returns:
        The straightness ratio (float).
    """
    start_node = route[0]
    end_node = route[-1]

    # Get coordinates of start and end nodes
    start_x = G.nodes[start_node]["x"]
    start_y = G.nodes[start_node]["y"]
    end_x = G.nodes[end_node]["x"]
    end_y = G.nodes[end_node]["y"]

    # Calculate straight-line distance (Euclidean distance)
    straight_line_distance = ox.distance.great_circle_vec(
        start_y, start_x, end_y, end_x
    )[0]  # Returns a numpy array, so take the first element

    # Calculate path length
    path_length = sum(
        ox.utils_graph.get_route_edge_attributes(G, route, attribute="length")
    )

    # Handle cases where path_length is zero to avoid division by zero
    if path_length == 0:
        return 0.0  # Or raise an exception, depending on your needs

    return straight_line_distance / path_length


def main():
    args = parse_args()
    place = args.location.strip()
    filename = f"./data/{place}_{args.type}.graphml"
    logging.info("Generating strightline for %s", place)
    ox.config(log_console=True)
    gdf = ox.geocode_to_gdf(place)
    if os.path.isfile(filename):
        G = load_cached_graph(filename)
    else:
        G = ox.graph_from_place(place, network_type=args.type, retain_all=False)
        cache_graph(G, filename)
    if type(gdf.geometry[0]) == MultiPolygon:
        exterior = gdf.geometry[0].geoms[0]
    else:
        exterior = gdf.geometry[0]
    corners_nodes = get_corners_nodes(G, exterior)
    closest_to_edge = work_out_edges(G, exterior, args.buffer)
    nodestoroute = closest_to_edge.union(corners_nodes)
    routes = calulate_routes(nodestoroute, G)
    routes.sort(key=lambda route: straightness_ratio(G, route), reverse=True)
    if args.np:
        display(routes[:3], G, gdf, nodestoroute, args.dont_display_edges, place)
    write_gpx(routes[:3], place, G)


def parse_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-l", "--location", help="Location to straight line")
    argParser.add_argument(
        "-e",
        "--dont-display-edges",
        help="Don't display the edge nodes we have found",
        action="store_false",
    )
    argParser.add_argument(
        "-t",
        "--type",
        choices=["all", "bike", "drive", "drive_service", "bike", "walk"],
        default="walk",
        help="What type of route, bike, drive, walk, all, drive_service",
    )
    argParser.add_argument(
        "-b",
        "--buffer",
        default=0.002,
        type=float,
        help="Distance from end for edge nodes",
    )
    argParser.add_argument(
        "--np",
        action="store_false",
    )
    return argParser.parse_args()


def close_to_edge(node, data, buffer):
    if Point(data["x"], data["y"]).within(buffer):
        return node
    else:
        return None


def get_corners_nodes(G, exterior) -> set:
    bounds = exterior.bounds
    corners = [
        (bounds[0], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
    ]
    cornernodes = set()
    for corner in corners:
        cornernodes.add(ox.nearest_nodes(G, corner[0], corner[1]))
    return cornernodes


def work_out_edges(G, exterior, edgebuffer) -> set:
    buffer = exterior.exterior.buffer(edgebuffer)
    closest_to_edge = set()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=mp.cpu_count() - 1
    ) as e:
        fut = [
            e.submit(close_to_edge, node, data, buffer)
            for node, data in G.nodes(data=True)
        ]
        for i, r in enumerate(concurrent.futures.as_completed(fut)):
            logging.info("Calculated close to edge %s total %s", i, len(fut))
            if r.result():
                closest_to_edge.add(r.result())
    logging.info(
        "Found the closest points to the edge count is %s", len(closest_to_edge)
    )
    if len(closest_to_edge) > 5:
        return closest_to_edge
    else:
        logging.info("Doubling buffer to find more at end")
        return work_out_edges(G, exterior, edgebuffer=edgebuffer * 2)


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
        with open(f"gpx/{place.strip().replace(' ', '_')}_{i}_route.gpx", "w") as f:
            f.write(gpx.to_xml())


if __name__ == "__main__":
    main()
