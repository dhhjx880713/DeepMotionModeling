# encoding: UTF-8

from morphablegraphs.utilities import get_semantic_motion_primitive_path, load_json_file
import numpy as np
import time
import logging
import argparse
from tabulate import tabulate
from copy import deepcopy
import mgrd as mgrd


test_matrix = {
    "n_samples": [10, 100, 500, 5000, 10000, 20000],
    "n_constraints_per_joint": [1, 2, 5],
    "n_different_joints": [1, 2, 5],
    "n_motions": [1]
}


class MotionPrimitiveEvaluator(object):

    def __init__(self, elementary_action, motion_primitive, data_repo):
        self.elementary_action = elementary_action
        self.motion_primitive = motion_primitive
        self.data_repo = data_repo
        self.skeleton = mgrd.SkeletonJSONLoader(r"../../mgrd/data/skeleton.json").load()
        self.motion_model = mgrd.MotionPrimitiveModel.load_from_file(
            self.skeleton,
            get_semantic_motion_primitive_path(self.elementary_action,
                                               self.motion_primitive,
                                               self.data_repo)
        )
        self.parameter_data = load_json_file(r"../../mgrd/data/parameter_matrix_data.json")
        self.test_joints = None

    def setup(self, n_samples, n_constraints_per_joint, n_different_joints, test_joints=None):
        self.n_constraints_per_joint = n_constraints_per_joint
        self.n_different_joints = n_different_joints
        if test_joints is not None:
            print('number of locomotion_synthesis_test joints is: ', )
            assert len(test_joints) == self.n_different_joints, ('The number of given locomotion_synthesis_test joints does not match the locomotion_synthesis_test'
                                                                 'number')
            self.test_joints = test_joints
        self.constraints = self._create_constraints()

    def _create_constraints(self):
        '''
        Create ramdom Cartesian constraints

        '''
        if self.test_joints is None:
            self.test_joints = self.parameter_data["test_joint_chains"][str(self.n_different_joints)]
        constraints = []
        random_sample = self.motion_model.get_random_samples(1)[0]
        quat_spline = self.motion_model.create_spatial_spline(random_sample)
        if self.n_constraints_per_joint == 1:
            frame_indices = [quat_spline.knots[-1]]
        elif self.n_constraints_per_joint == 2:
            frame_indices = [quat_spline.knots[0], quat_spline.knots[-1]]
        else:
            frame_indices = np.linspace(quat_spline.knots[0], quat_spline.knots[-1], self.n_constraints_per_joint)
        for joint_name in self.test_joints:
            joint = self.motion_model.skeleton.get_all_by_name([joint_name])
            cartesian_spline = quat_spline.to_cartesian(joint)
            for frame_idx in frame_indices:
                point = cartesian_spline.evaluate(frame_idx)
                constraints.append(mgrd.CartesianConstraint(point, joint_name, 1.0))
        return constraints

    def evaluate_cartesian_constraints(self, n_samples, n_constraints_per_joint, n_different_joints):
        samples = self.motion_model.get_random_samples(n_samples)
        quat_splines = self.motion_model.create_multiple_spatial_splines(samples)
        scores = mgrd.CartesianConstraint.score_splines(quat_splines, self.constraints)
        return min(scores)/(self.n_constraints_per_joint * self.n_different_joints)


def measure(func, run_count):
    perf = []
    for i in range(run_count):
        try:
            begin = time.clock()
            res = func()
            end = time.clock()
        except:
            logging.exception("Exception caught while measuring")
            continue
        perf.append(end - begin)
    return perf, res


def format_epilog(dict):
    epilog = "Available matrix parameters and default value sets:\n\n"
    epilog += tabulate(dict.items(), headers=["Name", "Value set"])
    return epilog


def run(repeats, argument_matrix, pipeline):
    initial_keystack = [i for i in argument_matrix.keys()]
    results = []
    assemble_and_run(initial_keystack.pop(), initial_keystack,
                     argument_matrix, {}, results, repeats, pipeline)
    return results


def assemble_and_run(cur_key, key_stack, repo_dict, arguments, results, repeats, pipeline):
    for val in repo_dict[cur_key]:
        arguments[cur_key] = val
        if len(key_stack) == 0:
            pipeline.setup(**arguments)
            f = lambda: pipeline.evaluate_cartesian_constraints(**arguments)
            perf, err = measure(f, repeats)
            arg_copy = deepcopy(arguments)
            n_runs = len(perf)
            if n_runs > 0:
                avgtime = sum(perf) / n_runs
            else:
                avgtime = -1
            results.append((arg_copy, avgtime, err, n_runs))
        else:
            next_key = key_stack.pop()
            assemble_and_run(next_key, key_stack, repo_dict,
                             arguments, results, repeats, pipeline)
            key_stack.append(next_key)


def pretty_results(results):
    """
    Pretty-Print a result table
    :param results: a list of results from run()
    """
    args, time, n_runs = results[0]
    heads = [name for name in args.keys()]

    # reshape it into something tabulate can cope with
    table = []
    for args, time, err, n_runs in results:
        values = [args[name] for name in heads]
        values.append(time)
        values.append(err)
        values.append(n_runs)
        table.append(values)
    heads.append("Time")
    heads.append("Distance $(cm^2)$")
    heads.append("#")

    print(tabulate(table, headers=heads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform matrix testing for MGRD",
                                     epilog=format_epilog(test_matrix),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--fix", nargs=2, action="append",
                        help="matrix parameter to keep constant")
    parser.add_argument("-n", "--runs", type=int, default=3,
                        help="number of executions to average over")

    args = parser.parse_args()
    if args.fix != None:
        for k, v in args.fix:
            print("Fixing {} to {}".format(k, v))
            # FIXME this assumes int matrix parameters, determine type from
            # test_matrix?
            test_matrix[k] = [int(v)]
    print("Testing, iterations per parameter set: " + str(args.runs))
    elementary_action = 'pickRight'
    motion_primitive = 'reach'
    data_repo = r'C:\repo'
    pipeline = MotionPrimitiveEvaluator(elementary_action,
                                        motion_primitive,
                                        data_repo)
    results = run(args.runs, test_matrix, pipeline)
    pretty_results(results)