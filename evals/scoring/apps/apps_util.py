# From APPS
import argparse
import faulthandler
import json
import os
import platform

# to run the solution files we're using a timing based approach
import signal
import sys

# used for debugging to time steps
from datetime import datetime
from enum import Enum

# for capturing the stdout
from io import StringIO
from typing import List

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import numpy as np
from ..utils.pyext2 import RuntimeModule


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def parse_args():
    parser = argparse.ArgumentParser(description="Utility for testing code generation.")
    parser.add_argument("-v", "--verbosity-level", action="store", type=int, help="")
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="leetcode",
        choices=[
            "leetcode",
            "atcoder",
            "codewars",
        ],
        help="which data source to gather from.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="question",
        choices=["question", "q", "solutions", "sol", "s", "starter", "tests", "t"],
        help="which type of data to receive.",
    )
    parser.add_argument(
        "-n", "--number", type=int, default=0, help="which problem to query."
    )

    args = parser.parse_args()
    return args


def get_valid_problems(data_dir="leetcode"):
    # these are unnecessary atm
    if data_dir == "leetcode":
        root = os.path.join(args.source, "data")
    elif data_dir == "atcoder":
        pass

    root = os.path.join(data_dir, "data")
    if os.path.exists(os.path.join(data_dir, "valid_problems.json")):
        with open(os.path.join(data_dir, "valid_problems.json"), "r") as f:
            return json.load(f)

    # after we compute it once let's save it and load that instead
    # TODO determine if might be better to reload each time
    tmp = os.listdir(root)
    valid_probs = []
    for folder in tmp:
        prob_path = os.path.join(root, folder)
        files = os.listdir(prob_path)
        # TODO add more validity checks
        if "input_output.json" in files or "sols.json" in files:
            valid_probs.append(prob_path)
    valid_probs = sorted(valid_probs)
    # with open(os.path.join(args.source,"valid_problems.json"), "w") as f:
    #    json.dump(valid_probs, f)
    return valid_probs


def get_question(problem_list, prob_index):
    root = problem_list[prob_index]
    # print("get q", root)
    if os.path.exists(os.path.join(root, "question.txt")):
        with open(os.path.join(root, "question.txt")) as f:
            question = f.readlines()
    else:
        print("question prompt not found")
        question = ""
    question = "".join(question)
    return question


def get_solutions(problem_list, prob_index):
    root = problem_list[prob_index]
    if os.path.exists(os.path.join(root, "solutions.json")):
        with open(os.path.join(root, "solutions.json")) as f:
            sols = json.load(f)
    return sols


def run_test(
    input_outputs=None,
    problem_list: List[str] = None,
    prob_index: int = None,
    test: str = None,
    debug: bool = False,
):
    """
    if test is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """

    if debug:
        print(f"start = {datetime.now().time()}")

    if problem_list is not None:
        root = problem_list[prob_index]  # noqa: F841

    in_outs = input_outputs
    if debug:
        print(f"test cases json = {in_outs['inputs']} {in_outs['outputs']}")

    if in_outs.get("fn_name") is None:
        which_type = CODE_TYPE.standard_input  # Standard input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based  # Call-based
        method_name = in_outs["fn_name"]
    if debug:
        print(f"loaded json = {datetime.now().time()}")

    if test is None:
        return in_outs
    elif test is not None:
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        results = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"  # noqa: E501
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:  # or True:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
                # print(f"{o}")
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except Exception:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            return results

        for index, inputs in enumerate(in_outs["inputs"]):
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except Exception:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index].items()}
                    ]
            except Exception:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index][0].items()}
                    ]
            except Exception:
                True

            if debug:
                print(
                    f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}"
                )
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    # print("------------")
                    # print(inputs)
                    output = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs["outputs"][index]
                    if (
                        isinstance(in_outs["outputs"][index], list)
                        and in_outs["outputs"][index]
                    ):
                        tmp_result = tmp_result or (
                            output == in_outs["outputs"][index][0]
                        )

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or (
                                [list(x) for x in output]
                                == in_outs["outputs"][index][0]
                            )
                    except Exception:
                        True
                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    print(
                        f"Standard input runtime error or time limit exceeded error = {e}"
                    )
                    results.append(-1)
                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(
                        f"outputs = {output}, test outputs = {in_outs['outputs'][index]},"
                        f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(
                            f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}"
                        )
                        results.append(-1)
                    signal.alarm(0)

                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]},"
                                f"inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)},"
                                f" {output == [in_outs['outputs'][index]]}"
                            )
                        else:
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]},"
                                f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                    continue

                if passed and debug:
                    print(
                        f"==> output = {output}, test outputs = {in_outs['outputs'][index]}"
                    )

                if custom_compare_(output, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or (
                                [e.strip() for e in output] == in_outs["outputs"][index]
                            )
                except Exception as e:
                    print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [
                            x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                        ]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(
                        filter(len, in_outs["outputs"][index])
                    )
                    in_outs["outputs"][index] = list(
                        map(lambda x: x.strip(), in_outs["outputs"][index])
                    )

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]},"
                            f"inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)},"
                            f" {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]},"
                            f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

                if tmp_result:
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    print(f"Failed check3 exception = {e}")
                    pass

                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs["outputs"][index]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float))
                        and np.allclose(output_float, gt_float)
                    )
                except Exception:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float))
                            and np.allclose(output_float, gt_float)
                        )
                except Exception:
                    pass

                if tmp_result:
                    results.append(tmp_result)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = output == in_outs["outputs"][index]
                except Exception as e:
                    print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = set(frozenset(s) for s in output) == set(
                        frozenset(s) for s in in_outs["outputs"][index]
                    )
                except Exception as e:
                    print(f"Failed check5 exception = {e}")

                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (
                        set(frozenset(round(float(t), 3) for t in s) for s in output)
                        == set(
                            frozenset(round(float(t), 3) for t in s)
                            for s in in_outs["outputs"][index]
                        )
                    )
                except Exception as e:
                    print(f"Failed check6 exception = {e}")

                if tmp_result and debug:
                    print("PASSED")

                results.append(tmp_result)

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]},"
                            f"inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)},"
                            f" {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]},"
                            f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

    return results


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    source: https://github.com/openai/human-eval
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def main(args):
    print(args)
    problem_list = sorted(get_valid_problems(args.source))
    prob_index = args.number

    # This checks it correctly loaded. remove this later
    assert prob_index < len(problem_list)

    if args.data == "q" or args.data == "question":
        tmp = get_question(problem_list, prob_index)
        print("q", tmp)
    elif args.data in [
        "solutions",
        "sol",
        "s",
    ]:
        tmp = get_solutions(problem_list, prob_index)
        print("sol", tmp)
    # TODO: fix this by adding get_starter
    # elif args.data == "starter":
    #     tmp = get_starter(problem_list, prob_index)
    #     print("starter", tmp)
    elif args.data in ["test", "t"]:
        # test it with sols
        sols = get_solutions(problem_list, prob_index)
        tmp = run_test(problem_list, prob_index, test=sols[0])

        print("results = ", tmp)
        print(
            "-2 = compile error, -1 is runtime error, False failed test, True passed test"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
