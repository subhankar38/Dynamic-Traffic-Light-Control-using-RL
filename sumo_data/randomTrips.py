#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import random
import bisect
import subprocess
from collections import defaultdict
import math
import argparse

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib  # noqa
from sumolib.miscutils import euclidean  # noqa
from sumolib.geomhelper import naviDegree, minAngleDegreeDiff  # noqa

DUAROUTER = sumolib.checkBinary('duarouter')

SOURCE_SUFFIX = ".src.xml"
SINK_SUFFIX = ".dst.xml"
VIA_SUFFIX = ".via.xml"


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Generate random trips for a SUMO network.")

    parser.add_argument("--net-file", required=True, help="Input SUMO network file (*.net.xml)")
    parser.add_argument("--output-file", required=True, help="Output trips file (*.rou.xml)")
    parser.add_argument("--trips", type=int, default=100, help="Number of trips to generate")
    parser.add_argument("--begin", type=int, default=0, help="Start time for trips")
    parser.add_argument("--end", type=int, default=3600, help="End time for trips")

class InvalidGenerator(Exception):
    pass
class RandomEdgeGenerator:

    def __init__(self, net, weight_fun):
        self.net = net
        self.weight_fun = weight_fun
        self.cumulative_weights = []
        self.total_weight = 0
        for edge in self.net._edges:
            self.total_weight += weight_fun(edge)
            self.cumulative_weights.append(self.total_weight)
        if self.total_weight == 0:
            raise InvalidGenerator()

    def get(self):
        r = random.random() * self.total_weight
        index = bisect.bisect(self.cumulative_weights, r)
        return self.net._edges[index]

    def write_weights(self, fname, interval_id, begin, end):
        # normalize to [0,100]
        normalizer = 100.0 / max(1, max(map(self.weight_fun, self.net._edges)))
        weights = [(self.weight_fun(e) * normalizer, e.getID()) for e in self.net.getEdges()]
        weights.sort(reverse=True)
        with open(fname, 'w+') as f:
            f.write('<edgedata>\n')
            f.write('    <interval id="%s" begin="%s" end="%s">\n' % (
                interval_id, begin, end))
            for weight, edgeID in weights:
                f.write('        <edge id="%s" value="%0.2f"/>\n' %
                        (edgeID, weight))
            f.write('    </interval>\n')
            f.write('</edgedata>\n')


class RandomTripGenerator:

    def __init__(self, source_generator, sink_generator, via_generator, intermediate, pedestrians):
        self.source_generator = source_generator
        self.sink_generator = sink_generator
        self.via_generator = via_generator
        self.intermediate = intermediate
        self.pedestrians = pedestrians

    def get_trip(self, min_distance, max_distance, maxtries=100, junctionTaz=False):
        for _ in range(maxtries):
            source_edge = self.source_generator.get()
            intermediate = [self.via_generator.get()
                            for i in range(self.intermediate)]
            sink_edge = self.sink_generator.get()
            if self.pedestrians:
                destCoord = sink_edge.getFromNode().getCoord()
            else:
                destCoord = sink_edge.getToNode().getCoord()

            coords = ([source_edge.getFromNode().getCoord()] +
                      [e.getFromNode().getCoord() for e in intermediate] +
                      [destCoord])
            distance = sum([euclidean(p, q)
                            for p, q in zip(coords[:-1], coords[1:])])
            if (distance >= min_distance
                    and (not junctionTaz or source_edge.getFromNode() != sink_edge.getToNode())
                    and (max_distance is None or distance < max_distance)):
                return source_edge, sink_edge, intermediate
        raise Exception("no trip found after %s tries" % maxtries)


def get_prob_fun(options, fringe_bonus, fringe_forbidden, max_length):
    # fringe_bonus None generates intermediate way points
    def edge_probability(edge):
        if options.vclass and not edge.allows(options.vclass):
            return 0  # not allowed
        if fringe_bonus is None and edge.is_fringe() and not options.pedestrians:
            return 0  # not suitable as intermediate way point
        if (fringe_forbidden is not None and edge.is_fringe(getattr(edge, fringe_forbidden)) and
                not options.pedestrians and
                (options.allow_fringe_min_length is None or edge.getLength() < options.allow_fringe_min_length)):
            return 0  # the wrong kind of fringe
        if (fringe_bonus is not None and options.viaEdgeTypes is not None and not edge.is_fringe() and
                edge.getType() in options.viaEdgeTypes):
            return 0  # the wrong type of edge (only allows depart and arrival on the fringe)
        prob = 1
        if options.length:
            if options.fringe_factor != 1.0 and fringe_bonus is not None and edge.is_fringe():
                # short fringe edges should not suffer a penalty
                prob *= max_length
            else:
                prob *= edge.getLength()
        if options.lanes:
            prob *= edge.getLaneNumber()
        prob *= (edge.getSpeed() ** options.speed_exponent)
        if (options.fringe_factor != 1.0 and
                not options.pedestrians and
                fringe_bonus is not None and
                edge.getSpeed() > options.fringe_threshold and
                edge.is_fringe(getattr(edge, fringe_bonus))):
            prob *= options.fringe_factor
        if options.edgeParam is not None:
            prob *= float(edge.getParam(options.edgeParam, 1.0))
        if options.angle_weight != 1.0 and fringe_bonus is not None:
            xmin, ymin, xmax, ymax = edge.getBoundingBox()
            ex, ey = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            nx, ny = options.angle_center
            edgeAngle = naviDegree(math.atan2(ey - ny, ex - nx))
            angleDiff = minAngleDegreeDiff(options.angle, edgeAngle)
            # print("e=%s nc=%s ec=%s ea=%s a=%s ad=%s" % (
            #    edge.getID(), options.angle_center, (ex,ey), edgeAngle,
            #    options.angle, angleDiff))
            # relDist = 2 * euclidean((ex, ey), options.angle_center) / max(xmax - xmin, ymax - ymin)
            # prob *= (relDist * (options.angle_weight - 1) + 1)
            if fringe_bonus == "_incoming":
                # source edge
                prob *= (angleDiff * (options.angle_weight - 1) + 1)
            else:
                prob *= ((180 - angleDiff) * (options.angle_weight - 1) + 1)

        return prob
    return edge_probability


class LoadedProps:

    def __init__(self, fname):
        self.weights = defaultdict(lambda: 0)
        for edge in sumolib.output.parse_fast(fname, 'edge', ['id', 'value']):
            self.weights[edge.id] = float(edge.value)

    def __call__(self, edge):
        return self.weights[edge.getID()]


def buildTripGenerator(net, options):
    try:
        max_length = 0
        for edge in net.getEdges():
            if not edge.is_fringe():
                max_length = max(max_length, edge.getLength())
        forbidden_source_fringe = None if options.allow_fringe else "_outgoing"
        forbidden_sink_fringe = None if options.allow_fringe else "_incoming"
        source_generator = RandomEdgeGenerator(
            net, get_prob_fun(options, "_incoming", forbidden_source_fringe, max_length))
        sink_generator = RandomEdgeGenerator(
            net, get_prob_fun(options, "_outgoing", forbidden_sink_fringe, max_length))
        if options.weightsprefix:
            if os.path.isfile(options.weightsprefix + SOURCE_SUFFIX):
                source_generator = RandomEdgeGenerator(
                    net, LoadedProps(options.weightsprefix + SOURCE_SUFFIX))
            if os.path.isfile(options.weightsprefix + SINK_SUFFIX):
                sink_generator = RandomEdgeGenerator(
                    net, LoadedProps(options.weightsprefix + SINK_SUFFIX))
    except InvalidGenerator:
        print("Error: no valid edges for generating source or destination. Try using option --allow-fringe",
              file=sys.stderr)
        return None

    try:
        via_generator = RandomEdgeGenerator(
            net, get_prob_fun(options, None, None, 1))
        if options.weightsprefix and os.path.isfile(options.weightsprefix + VIA_SUFFIX):
            via_generator = RandomEdgeGenerator(
                net, LoadedProps(options.weightsprefix + VIA_SUFFIX))
    except InvalidGenerator:
        if options.intermediate > 0:
            print(
                "Error: no valid edges for generating intermediate points", file=sys.stderr)
            return None
        else:
            via_generator = None

    return RandomTripGenerator(
        source_generator, sink_generator, via_generator, options.intermediate, options.pedestrians)


def is_walk_attribute(attr):
    for cand in ['arrivalPos', 'speed=', 'duration=', 'busStop=']:
        if cand in attr:
            return True
    return False
def is_persontrip_attribute(attr):
    for cand in ['vTypes', 'modes']:
        if cand in attr:
            return True
    return False
def is_person_attribute(attr):
    for cand in ['departPos', 'type']:
        if cand in attr:
            return True
    return False
def is_vehicle_attribute(attr):
    for cand in ['depart', 'arrival', 'line', 'Number', 'type']:
        if cand in attr:
            return True
    return False
def split_trip_attributes(tripattrs, pedestrians, hasType):
    # handle attribute values with a space
    # assume that no attribute value includes an '=' sign
    allattrs = []
    for a in tripattrs.split():
        if "=" in a:
            allattrs.append(a)
        else:
            if len(allattrs) == 0:
                print("Warning: invalid trip-attribute '%s'" % a)
            else:
                allattrs[-1] += ' ' + a
    # figure out which of the tripattrs belong to the <person> or <vehicle>,
    # which belong to the <vType> and which belong to the <walk> or <persontrip>
    vehicleattrs = []
    personattrs = []
    vtypeattrs = []
    otherattrs = []
    for a in allattrs:
        if pedestrians:
            if is_walk_attribute(a) or is_persontrip_attribute(a):
                otherattrs.append(a)
            elif is_person_attribute(a):
                personattrs.append(a)
            else:
                vtypeattrs.append(a)
        else:
            if is_vehicle_attribute(a):
                vehicleattrs.append(a)
            else:
                vtypeattrs.append(a)

    if not hasType:
        if pedestrians:
            personattrs += vtypeattrs
        else:
            vehicleattrs += vtypeattrs
        vtypeattrs = []

    return (prependSpace(' '.join(vtypeattrs)),
            prependSpace(' '.join(vehicleattrs)),
            prependSpace(' '.join(personattrs)),
            prependSpace(' '.join(otherattrs)))


def prependSpace(s):
    if len(s) == 0 or s[0] == " ":
        return s
    else:
        return " " + s


def main(options):
    net = sumolib.net.readNet(options.netfile)
    if options.min_distance > net.getBBoxDiameter() * (options.intermediate + 1):
        options.intermediate = int(
            math.ceil(options.min_distance / net.getBBoxDiameter())) - 1
        print(("Warning: setting number of intermediate waypoints to %s to achieve a minimum trip length of " +
               "%s in a network with diameter %.2f.") % (
            options.intermediate, options.min_distance, net.getBBoxDiameter()))

    if options.angle_weight != 1:
        xmin, ymin, xmax, ymax = net.getBoundary()
        options.angle_center = (xmin + xmax) / 2, (ymin + ymax) / 2

    trip_generator = buildTripGenerator(net, options)
    idx = 0

    vtypeattrs, options.tripattrs, personattrs, otherattrs = split_trip_attributes(
        options.tripattrs, options.pedestrians, options.vehicle_class)

    vias = {}

    def generate_one(idx):
        label = "%s%s" % (options.tripprefix, idx)
        try:
            source_edge, sink_edge, intermediate = trip_generator.get_trip(
                options.min_distance, options.max_distance, options.maxtries,
                options.junctionTaz)
            combined_attrs = options.tripattrs
            if options.fringeattrs and source_edge.is_fringe(source_edge._incoming):
                combined_attrs += " " + options.fringeattrs
            if options.junctionTaz:
                attrFrom = ' fromJunction="%s"' % source_edge.getFromNode().getID()
                attrTo = ' toJunction="%s"' % sink_edge.getToNode().getID()
            else:
                attrFrom = ' from="%s"' % source_edge.getID()
                attrTo = ' to="%s"' % sink_edge.getID()
            via = ""
            if len(intermediate) > 0:
                via = ' via="%s" ' % ' '.join(
                    [e.getID() for e in intermediate])
                if options.validate:
                    vias[label] = via
            if options.pedestrians:
                fouttrips.write(
                    '    <person id="%s" depart="%.2f"%s>\n' % (label, depart, personattrs))
                if options.persontrips:
                    fouttrips.write(
                        '        <personTrip%s%s%s/>\n' % (attrFrom, attrTo, otherattrs))
                elif options.personrides:
                    fouttrips.write(
                        '        <ride from="%s" to="%s" lines="%s"%s/>\n' % (
                            source_edge.getID(), sink_edge.getID(), options.personrides, otherattrs))
                else:
                    fouttrips.write(
                        '        <walk%s%s%s/>\n' % (attrFrom, attrTo, otherattrs))
                fouttrips.write('    </person>\n')
            else:
                if options.jtrrouter:
                    attrTo = ''
                combined_attrs = attrFrom + attrTo + via + combined_attrs
                if options.flows > 0:
                    if options.binomial:
                        for j in range(options.binomial):
                            fouttrips.write(('    <flow id="%s#%s" begin="%s" end="%s" probability="%s"%s/>\n') % (
                                label, j, options.begin, options.end, 1.0 / options.period / options.binomial,
                                combined_attrs))
                    else:
                        fouttrips.write(('    <flow id="%s" begin="%s" end="%s" period="%s"%s/>\n') % (
                            label, options.begin, options.end, options.period * options.flows, combined_attrs))
                else:
                    fouttrips.write('    <trip id="%s" depart="%.2f"%s/>\n' % (
                        label, depart, combined_attrs))
        except Exception as exc:
            print(exc, file=sys.stderr)
        return idx + 1

    with open(options.tripfile, 'w') as fouttrips:
        sumolib.writeXMLHeader(fouttrips, "$Id$", "routes")  # noqa
        if options.vehicle_class:
            vTypeDef = '    <vType id="%s" vClass="%s"%s/>\n' % (
                options.vtypeID, options.vehicle_class, vtypeattrs)
            if options.vtypeout:
                # ensure that trip output does not contain types, file may be
                # overwritten by later call to duarouter
                if options.additional is None:
                    options.additional = options.vtypeout
                else:
                    options.additional += ",options.vtypeout"
                with open(options.vtypeout, 'w') as fouttype:
                    sumolib.writeXMLHeader(fouttype, "$Id$", "additional")  # noqa
                    fouttype.write(vTypeDef)
                    fouttype.write("</additional>\n")
            else:
                fouttrips.write(vTypeDef)
            options.tripattrs += ' type="%s"' % options.vtypeID
            personattrs += ' type="%s"' % options.vtypeID
        depart = sumolib.miscutils.parseTime(options.begin)
        maxTime = sumolib.miscutils.parseTime(options.end)
        if trip_generator:
            if options.flows == 0:
                while depart < maxTime:
                    if options.binomial is None:
                        # generate with constant spacing
                        idx = generate_one(idx)
                        depart += options.period
                    else:
                        # draw n times from a Bernoulli distribution
                        # for an average arrival rate of 1 / period
                        prob = 1.0 / options.period / options.binomial
                        for _ in range(options.binomial):
                            if random.random() < prob:
                                idx = generate_one(idx)
                        depart += 1
            else:
                for _ in range(options.flows):
                    idx = generate_one(idx)

        fouttrips.write("</routes>\n")

    # call duarouter for routes or validated trips
    args = [DUAROUTER, '-n', options.netfile, '-r', options.tripfile, '--ignore-errors',
            '--begin', str(options.begin), '--end', str(options.end), '--no-step-log']
    if options.additional is not None:
        args += ['--additional-files', options.additional]
    if options.carWalkMode is not None:
        args += ['--persontrip.transfer.car-walk', options.carWalkMode]
    if options.walkfactor is not None:
        args += ['--persontrip.walkfactor', options.walkfactor]
    if options.remove_loops:
        args += ['--remove-loops']
    if options.vtypeout is not None:
        args += ['--vtype-output', options.vtypeout]
    if options.junctionTaz:
        args += ['--junction-taz']
    if not options.verbose:
        args += ['--no-warnings']
    else:
        args += ['-v']

    if options.routefile:
        args2 = args + ['-o', options.routefile]
        print("calling", " ".join(args2))
        sys.stdout.flush()
        subprocess.call(args2)
        sys.stdout.flush()

    if options.validate:
        # write to temporary file because the input is read incrementally
        tmpTrips = options.tripfile + ".tmp"
        args2 = args + ['-o', tmpTrips, '--write-trips']
        if options.junctionTaz:
            args2 += ['--write-trips.junctions']
        print("calling", " ".join(args2))
        sys.stdout.flush()
        subprocess.call(args2)
        sys.stdout.flush()
        os.remove(options.tripfile)  # on windows, rename does not overwrite
        os.rename(tmpTrips, options.tripfile)

    if options.weights_outprefix:
        idPrefix = ""
        if options.tripprefix:
            idPrefix = options.tripprefix + "."
        trip_generator.source_generator.write_weights(
            options.weights_outprefix + SOURCE_SUFFIX,
            idPrefix + "src", options.begin, options.end)
        trip_generator.sink_generator.write_weights(
            options.weights_outprefix + SINK_SUFFIX,
            idPrefix + "dst", options.begin, options.end)
        if trip_generator.via_generator:
            trip_generator.via_generator.write_weights(
                options.weights_outprefix + VIA_SUFFIX,
                idPrefix + "via", options.begin, options.end)

    # return wether trips could be generated as requested
    return trip_generator is not None


if __name__ == "__main__":
    if not main(get_options()):
        sys.exit(1)
