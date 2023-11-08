# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from threading import Event, Thread
import psutil, time, os
from datetime import timedelta
import numpy as np

def print_perf_counters(perf_counts_list):
    max_print_length = 20
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = timedelta()
        total_time_cpu = timedelta()
        print(f"Performance counts for {ni}-th infer request")
        for pi in perf_counts:
            print(f"{pi.node_name[:max_print_length - 4] + '...' if (len(pi.node_name) >= max_print_length) else pi.node_name:<20} "
                f"{str(pi.status):<20} "
                f"layerType: {pi.node_type[:max_print_length - 4] + '...' if (len(pi.node_type) >= max_print_length) else pi.node_type:<20} "
                f"execType: {pi.exec_type:<20} "
                f"realTime (ms): {pi.real_time / timedelta(milliseconds=1):<10.3f} "
                f"cpuTime (ms): {pi.cpu_time / timedelta(milliseconds=1):<10.3f}")

            total_time += pi.real_time
            total_time_cpu += pi.cpu_time
        print(f'Total time:     {total_time / timedelta(milliseconds=1)} milliseconds')
        print(f'Total CPU time: {total_time_cpu / timedelta(milliseconds=1)} milliseconds\n')

def print_detail_result(result_list):
    """ Print_perf_counters_sort result
    """
    max_print_length = 20
    for tmp_result in result_list:
        node_name = tmp_result[0]
        layerStatus = tmp_result[1]
        layerType = tmp_result[2]
        real_time = tmp_result[3]
        cpu_time = tmp_result[4]
        real_proportion = "%.2f" % (tmp_result[5] * 100)
        if real_proportion == "0.00":
            real_proportion = "N/A"
        execType = tmp_result[6]
        print(f"{node_name[:max_print_length - 4] + '...' if (len(node_name) >= max_print_length) else node_name:<20} "
                f"{str(layerStatus):<20} "
                f"layerType: {layerType[:max_print_length - 4] + '...' if (len(layerType) >= max_print_length) else layerType:<20} "
                f"execType: {execType:<20} "
                f"realTime (ms): {real_time / 1000:<10.3f} "
                f"cpuTime (ms): {cpu_time / 1000:<10.3f}"
                f"proportion: {str(real_proportion +'%'):<8}")

def print_perf_counters_sort(perf_counts_list,sort_flag="sort"):
    """ Print opts time cost and can be sorted according by each opts time cost
    """
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = timedelta()
        total_time_cpu = timedelta()
        print(f"Performance counts sorted for {ni}-th infer request")
        for pi in perf_counts:
            total_time += pi.real_time
            total_time_cpu += pi.cpu_time

        total_time = total_time.microseconds
        total_time_cpu = total_time_cpu.microseconds
        total_real_time_proportion = 0
        total_detail_data=[]
        for pi in perf_counts:
            node_name = pi.node_name
            layerStatus = pi.status
            layerType = pi.node_type
            real_time = pi.real_time.microseconds
            cpu_time = pi.cpu_time.microseconds
            real_proportion = round(real_time/total_time,4)
            execType = pi.exec_type
            tmp_data=[node_name,layerStatus,layerType,real_time,cpu_time,real_proportion,execType]
            total_detail_data.append(tmp_data)
            total_real_time_proportion += real_proportion
        total_detail_data = np.array(total_detail_data)
        if sort_flag=="sort":
            total_detail_data = sorted(total_detail_data,key=lambda tmp_data:tmp_data[-4],reverse=True)
        elif sort_flag=="no_sort":
            total_detail_data = total_detail_data
        elif sort_flag=="simple_sort":
            total_detail_data = sorted(total_detail_data,key=lambda tmp_data:tmp_data[-4],reverse=True)
            total_detail_data = [tmp_data for tmp_data in total_detail_data if str(tmp_data[1])!="Status.NOT_RUN"]
        print_detail_result(total_detail_data)
        print(f'Total time:       {total_time / 1000:.3f} milliseconds')
        print(f'Total CPU time:   {total_time_cpu / 1000:.3f} milliseconds')
        print(f'Total proportion: {"%.2f"%(round(total_real_time_proportion)*100)} % \n')
    return total_detail_data

'''
class MemConsumption:
    def __init__(self):
        """Initialize MemConsumption."""
        self.g_exit_get_mem_thread = False
        self.g_end_collect_mem = False
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1
        self.g_event = Event()
        self.g_data_event = Event()

    def collect_memory_consumption(self):
        """Collect the data."""
        while self.g_exit_get_mem_thread is False:
            self.g_event.wait()
            while True:
                process = psutil.Process(os.getpid())
                rss_mem_data = process.memory_info().rss / float(2**20)
                try:
                    shared_mem_data = process.memory_info().shared / float(2**20)
                except Exception:
                    shared_mem_data = -1
                if rss_mem_data > self.g_max_rss_mem_consumption:
                    self.g_max_rss_mem_consumption = rss_mem_data
                if shared_mem_data > self.g_max_shared_mem_consumption:
                    self.g_max_shared_mem_consumption = shared_mem_data
                self.g_data_event.set()
                if self.g_end_collect_mem is True:
                    self.g_event.set()
                    self.g_event.clear()
                    self.g_end_collect_mem = False
                    break
                time.sleep(500 / 1000)

    def start_collect_memory_consumption(self):
        """Start collect."""
        self.g_end_collect_mem = False
        self.g_event.set()

    def end_collect_momory_consumption(self):
        """Stop collect."""
        self.g_end_collect_mem = True
        self.g_event.wait()

    def get_max_memory_consumption(self):
        """Return the data."""
        self.g_data_event.wait()
        self.g_data_event.clear()
        return self.g_max_rss_mem_consumption, self.g_max_shared_mem_consumption

    def clear_max_memory_consumption(self):
        """Clear MemConsumption."""
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1

    def start_collect_mem_consumption_thread(self):
        """Start the thread."""
        self.t_mem_thread = Thread(target=self.collect_memory_consumption)
        self.t_mem_thread.start()

    def end_collect_mem_consumption_thread(self):
        """End the thread."""
        self.g_event.set()
        self.g_data_event.set()
        self.g_end_collect_mem = True
        self.g_exit_get_mem_thread = True
        self.t_mem_thread.join()
'''
