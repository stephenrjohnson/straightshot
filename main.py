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
from shapely.geometry import MultiPolygon, Point, Polygon


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count() - 1) as e:
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


def main():
    args = parse_args()
    place = args.location.strip()
    filename = f"./data/${place}_{args.type}.graphml"
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
    routes.sort(
        key=lambda route: sum(
            ox.utils_graph.get_route_edge_attributes(G, route, attribute="length")
        ),
        reverse=True,
    )
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
        choices=["all", "bike", "drive", "drive_service", "bike"],
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
    bounds =  exterior.bounds
    corners = [(bounds[0],bounds[1]),(bounds[0],bounds[3]),(bounds[1],bounds[0]),(bounds[1],bounds[1])]
    cornernodes = set()
    for corner in corners:
        cornernodes.add(ox.nearest_nodes(G, corner[0], corner[1]))
    return cornernodes


def work_out_edges(G, exterior, edgebuffer) -> set:
    # Find the closest nodes to the boundary of the place
    # closest_to_edge = set(ox.nearest_nodes(G, *exterior.exterior.coords.xy))
    buffer = exterior.exterior.buffer(edgebuffer)
    closest_to_edge = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count() - 1) as e:
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
        with open(f"gpx/{place.strip().replace(' ','_')}_{i}_route.gpx", "w") as f:
            f.write(gpx.to_xml())


if __name__ == "__main__":
    main()
