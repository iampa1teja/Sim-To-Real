"""A fixed terminal table for passive joint telemetry; standard library only."""
import math
import os
import shutil
import sys


class JointAngleTable:
    """Keep a table below ordinary console output and update changed cells only."""

    WIDTHS = (14, 9, 9, 9, 7)
    TITLE = "Joint angles (deg)   S=still M=moving L=limit B=|gap|>180"

    def __init__(self, joint_order, *, stream=None, terminal_size=None):
        self.order = tuple(joint_order)
        self.stream = sys.stdout if stream is None else stream
        self.terminal_size = terminal_size or shutil.get_terminal_size
        self.interactive = bool(self.stream.isatty() and os.environ.get("TERM") != "dumb")
        self.active = False
        self.size = None
        self.cells = None
        self.row_positions = {}

    @staticmethod
    def _number(value):
        if value is None or not math.isfinite(float(value)):
            return "--".rjust(9)
        text = f"{float(value):+9.2f}"
        return text if len(text) <= 9 else f"{float(value):+9.1e}"

    def _cells(self, row):
        real = row.get("real_deg", row.get("mapped_follower_degrees"))
        sim = row.get("sim_deg", row.get("sim_physx_degrees"))
        gaps = row.get("sim_minus_real_deg", row.get("sim_minus_mapped_follower_degrees"))
        static = row.get("static")
        limits = row.get("at_limit", row.get("at_runtime_limit", [False] * len(self.order)))
        branches = row.get("suspected_branch_error", [abs(gap) > 180 for gap in gaps])
        values = {}
        for i, name in enumerate(self.order):
            flags = ["?" if static is None else "S" if static[i] else "M"]
            if limits[i]:
                flags.append("L")
            if branches[i]:
                flags.append("B")
            values[name] = (self._number(real[i]), self._number(sim[i]), self._number(gaps[i]),
                            " ".join(flags).ljust(self.WIDTHS[-1]))
        return values

    def _lines(self, cells):
        border = "+" + "+".join("-" * (width + 2) for width in self.WIDTHS) + "+"

        def line(values):
            return "| " + " | ".join(value.ljust(width) for value, width in zip(values, self.WIDTHS)) + " |"

        lines = [self.TITLE, border, line(("Joint", "Real", "Sim", "Sim-real", "Flags")), border]
        names = [name for name in self.order if name != "gripper"]
        if "gripper" in self.order:
            names.append("gripper")
        for name in names:
            if name == "gripper" and len(lines) > 4:
                lines.append(border)
            self.row_positions[name] = len(lines)
            lines.append(line((name[:self.WIDTHS[0]], *cells[name])))
        lines.append(border)
        return lines

    def update(self, row):
        """Display one observation without printing new rows on an ANSI terminal."""
        cells = self._cells(row)
        lines = self._lines(cells)
        columns, rows = self.terminal_size()
        if not self.interactive or columns < len(lines[1]) or rows < len(lines) + 2:
            self.close()
            self.stream.write("\n".join(lines) + "\n")
            self.stream.flush()
            return
        size = columns, rows
        top = rows - len(lines) + 1
        if not self.active or self.size != size:
            # Reserve bottom rows; subsequent application messages and input
            # scroll in the area above the table. No alternate screen is used.
            self.stream.write("\033[r" + f"\033[{rows};1H" + "\n" * len(lines)
                              + f"\033[1;{top - 1}r"
                              + "".join(f"\033[{top + i};1H\033[2K{line}" for i, line in enumerate(lines))
                              + f"\033[{top - 1};1H")
            self.active, self.size, self.cells = True, size, cells
        else:
            updates = []
            for name, values in cells.items():
                column = self.WIDTHS[0] + 6
                for i, value in enumerate(values):
                    if value != self.cells[name][i]:
                        updates.append(f"\033[{top + self.row_positions[name]};{column}H{value}")
                    column += self.WIDTHS[i + 1] + 3
            if updates:
                # Save/restore the input cursor so typing commands remains usable.
                self.stream.write("\0337" + "".join(updates) + "\0338")
            self.cells = cells
        self.stream.flush()

    def close(self):
        """Restore normal terminal scrolling and leave the final table visible."""
        if self.active:
            self.stream.write(f"\033[r\033[{self.size[1]};1H\n")
            self.stream.flush()
            self.active = False
