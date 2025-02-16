import time
class Profiler:
    def __init__(self):
        self.stats = {
            "run_time" : {},
            "calls" : {},
            "mean_run_time" : {},
        }
        
    def time_it(self, func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            _class_name = type(args[0]).__name__ if args else "Unknown"
            _func_name = func.__name__+f" || Class : {_class_name}"
            _time = end-start
            self._update_stats(_func_name, _time)
            return result
        return wrapper            

    def _update_stats(self,_func_name, _time):
        if _func_name not in self.stats["run_time"]:
            self.stats["run_time"][_func_name] = []
        else:
            self.stats["run_time"][_func_name].append(_time)       

        if _func_name not in self.stats["calls"]: 
            self.stats["calls"][_func_name] = 0 
        self.stats["calls"][_func_name] += 1
            
        if _func_name not in self.stats["mean_run_time"]:
            self.stats["mean_run_time"][_func_name] = 0.0
        self.stats["mean_run_time"][_func_name] = sum(self.stats["run_time"][_func_name])/self.stats["calls"][_func_name]
            
    def generate_performance_log(self, file_name="performance.log"):
        # ANSI color codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

        with open(file_name, "w") as f:
            for func_name in self.stats["run_time"]:
                times = self.stats["run_time"][func_name]
                call_count = self.stats["calls"].get(func_name, 0)
                mean_time = self.stats["mean_run_time"].get(func_name, 0)

                f.write(f"{GREEN}Function: {func_name}{RESET}\n")
                f.write(f"  {YELLOW}Total Time:{RESET} {','.join([str(t) for t in times])} seconds\n")
                f.write(f"  {YELLOW}Call Count:{RESET} {call_count}\n")
                f.write(f"  {RED}Mean Time:{RESET} {mean_time:.6f} seconds\n")
                f.write("-" * 40 + "\n") 