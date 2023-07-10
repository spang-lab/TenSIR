import datetime as dt
from os.path import join


def format_time(seconds):
    if seconds < 60:
        return f"{seconds}"


def main():
    results = {}
    for sampling in ("hmc", "mh"):
        results[sampling] = []

        for m in range(3, 9):

            month_durations = []
            for run in range(10):
                log_path = join("logs", sampling, f"{m:02d}", f"{sampling}-{m:02d}-{run:02d}.log")
                with open(log_path, "r") as f:
                    lines = f.readlines()

                start = dt.datetime.strptime(lines[0].split(",")[0].replace("[", ""), "%Y-%m-%d %H:%M:%S")
                end = dt.datetime.strptime(lines[-1].split(",")[0].replace("[", ""), "%Y-%m-%d %H:%M:%S")

                month_durations.append((end - start).total_seconds())

            avg = sum(month_durations) / len(month_durations)

            results[sampling].append((m, avg))

    for sampling, times in results.items():
        print(sampling)
        for m, time in times:
            print(m, "   ", f"${time:.1e}$".replace("e+0", "\\cdot 10^"))

    print(results)


if __name__ == "__main__":
    main()
