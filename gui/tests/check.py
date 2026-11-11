# Copyright (C) 2026 Yves Renard, Konstantinos Poulios.
#
# This file is a part of GetFEM
#
# GetFEM  is  free software;  you  can  redistribute  it  and/or modify it
# under  the  terms  of the  GNU  Lesser General Public License as published
# by  the  Free Software Foundation;  either version 3 of the License,  or
# (at your option) any later version along with the GCC Runtime Library
# Exception either version 3.1 or (at your option) any later version.
# This program  is  distributed  in  the  hope  that it will be useful,  but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or  FITNESS  FOR  A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License and GCC Runtime Library Exception for more details.
# You  should  have received a copy of the GNU Lesser General Public License
# along  with  this program. If not, see https://www.gnu.org/licenses/.

"""Runs one test file of the interface for `make check` (gui/Makefile.am):

    python tests/check.py tests/test_heat.py

with pytest, without its cache (the sources may be read-only). Exit status
for the test harness of Automake: 0 passed, 1 failed, 77 skipped (pytest
not installed, or every test of the file skipped: PySide6, PyVista, Gmsh...
missing).
"""

import sys
from pathlib import Path


class _Count:
    """A pytest plugin: the outcome of the tests."""

    def __init__(self):
        self.passed = self.skipped = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1
        elif report.skipped:
            self.skipped += 1

    def pytest_collectreport(self, report):
        if report.skipped:                     # skipped as a whole (allow_module_level)
            self.skipped += 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        import pytest
    except ImportError:
        print("pytest is not installed: the tests of the interface are skipped")
        return 77
    path = Path(argv[1]).resolve()
    count = _Count()
    code = pytest.main([str(path), "-q", "-p", "no:cacheprovider", "--rootdir", str(path.parent)],
                       plugins=[count])
    if code not in (0, 5):                     # 5: no test collected
        return 1
    return 0 if count.passed else 77


if __name__ == "__main__":
    sys.exit(main(sys.argv))
