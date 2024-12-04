#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd

def calculate_mean(time_points: pd.Series, values: pd.Series) -> float:
    prev_time_point = time_points.iloc[0]
    delay_sum = 0.0
    value_sum = 0.0

    for time_point, value in zip(time_points.iloc[1:], values):
        delay = time_point - prev_time_point
        prev_time_point = time_point
        delay_sum += delay
        value_sum += value * delay

    try:
        res = value_sum / delay_sum
        return res
    except ZeroDivisionError:
        return 0

def calculate_std_dev(time_points: pd.Series, values: pd.Series, mean: float) -> float:
    prev_time_point = time_points.iloc[0]
    delay_sum = 0.0
    value_sum = 0.0

    for time_point, value in zip(time_points.iloc[1:], values):
        delay = time_point - prev_time_point
        prev_time_point = time_point
        delay_sum += delay
        value_sum += ((value - mean) ** 2) * delay

    try:
        res = np.sqrt(value_sum / delay_sum)
        return res
    except ZeroDivisionError:
        return 0


# In[22]:


import os
from pathlib import Path
import pandas as pd
import attr
from typing import Optional

verification_data_dir = Path('./verification_data')

@attr.frozen
class ParamsData:
    common_props: pd.DataFrame
    time_wait_allocate: pd.DataFrame
    time_in_system: pd.DataFrame

datas: list[ParamsData] = []
params_mat: list[pd.Series] = []

for dirpath, dir, filenames in os.walk(verification_data_dir):
    dir_path = Path(dirpath)
    if dir_path.name == verification_data_dir.name:
        continue
    params = tuple(int(n) for n in dir_path.name.split('_'))

    params_mat.append(pd.Series({
        'Кількість сторінок': params[0],
        'Кількість процесорів': params[1],
        'Кількість дисків':params[2],
        'Початок сторінок':params[3],
        'Кінець сторінок': params[4],
        'Середій інтервал надходження завдань': params[5]
    }))

    common_props: Optional[pd.DataFrame] = None
    time_wait_allocate: Optional[pd.DataFrame] = None
    time_in_system: Optional[pd.DataFrame] = None
    for file_name in filenames:
        data = pd.read_csv(Path(dirpath) / file_name)
        if file_name.startswith('commonProps'):
            common_props = data
            # threshold = 0.01
            # common_props['processorsLoad'] = common_props['processorsLoad'].apply(lambda x: 0 if x < threshold else x)
            # common_props['diskLoad'] = common_props['diskLoad'].apply(lambda x: 0 if x < threshold else x)
            # common_props['ioChannelLoad'] = common_props['ioChannelLoad'].apply(lambda x: 0 if x < threshold else x)
        elif file_name.startswith('timeWaitAllocate'):
            time_wait_allocate = data
        elif file_name.startswith('timeInSystem'):
            time_in_system = data
    
    if common_props is not None and time_wait_allocate is not None and time_in_system is not None:
        datas.append(ParamsData(common_props, time_wait_allocate, time_in_system))
    else:
        raise Exception('empty data')


# In[23]:


params_data_frame = pd.concat(params_mat, axis=1)
params_data_frame = params_data_frame.T
params_data_frame.to_csv(verification_res_dir_path / 'params.csv', index=True, index_label='Індекс')
params_data_frame


# In[24]:


from array import array

@attr.frozen
class MeanStddevStats:
    diskLoad_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    diskLoad_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))
    ioChannelLoad_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    ioChannelLoad_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))
    processorsLoad_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    processorsLoad_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))
    totalWaitAllocate_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    totalWaitAllocate_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))
    useOfPage_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    useOfPage_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))
    timeInSystem_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    timeInSystem_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))
    timeWaitAllocate_mean: array[float] = attr.field(init=False, factory=lambda: array('d'))
    timeWaitAllocate_std_dev: array[float] = attr.field(init=False, factory=lambda: array('d'))

mean_stddev_stats_list: list[pd.DataFrame] = []

for index, params_data in enumerate(datas):
    mean_stddev_stats = MeanStddevStats()

    for run_num, group in params_data.common_props.groupby('runNumber'):
        # Calculate means and standard deviations
        diskLoad_mean = calculate_mean(group['timePoint'], group['diskLoad'])
        diskLoad_std_dev = calculate_std_dev(group['timePoint'], group['diskLoad'], diskLoad_mean)

        ioChannelLoad_mean = calculate_mean(group['timePoint'], group['ioChannelLoad'])
        ioChannelLoad_std_dev = calculate_std_dev(group['timePoint'], group['ioChannelLoad'], ioChannelLoad_mean)

        processorsLoad_mean = calculate_mean(group['timePoint'], group['processorsLoad'])
        processorsLoad_std_dev = calculate_std_dev(group['timePoint'], group['processorsLoad'], processorsLoad_mean)

        totalWaitAllocate_mean = calculate_mean(group['timePoint'], group['totalWaitAllocate'])
        totalWaitAllocate_std_dev = calculate_std_dev(group['timePoint'], group['totalWaitAllocate'], totalWaitAllocate_mean)

        useOfPage_mean = calculate_mean(group['timePoint'], group['useOfPage'])
        useOfPage_std_dev = calculate_std_dev(group['timePoint'], group['useOfPage'], useOfPage_mean)

        mean_stddev_stats.diskLoad_mean.append(diskLoad_mean)
        mean_stddev_stats.diskLoad_std_dev.append(diskLoad_std_dev)

        mean_stddev_stats.ioChannelLoad_mean.append(ioChannelLoad_mean)
        mean_stddev_stats.ioChannelLoad_std_dev.append(ioChannelLoad_std_dev)

        mean_stddev_stats.processorsLoad_mean.append(processorsLoad_mean)
        mean_stddev_stats.processorsLoad_std_dev.append(processorsLoad_std_dev)

        mean_stddev_stats.totalWaitAllocate_mean.append(totalWaitAllocate_mean)
        mean_stddev_stats.totalWaitAllocate_std_dev.append(totalWaitAllocate_std_dev)

        mean_stddev_stats.useOfPage_mean.append(useOfPage_mean)
        mean_stddev_stats.useOfPage_std_dev.append(useOfPage_std_dev)
    
    for run_num, group in params_data.time_in_system.groupby('runNumber'):
        timeInSystem_mean = calculate_mean(group['timePoint'], group['timeInSystem'])
        timeInSystem_std_dev = calculate_std_dev(group['timePoint'], group['timeInSystem'], timeInSystem_mean)
        mean_stddev_stats.timeInSystem_mean.append(timeInSystem_mean)
        mean_stddev_stats.timeInSystem_std_dev.append(timeInSystem_std_dev)

    for run_num, group in params_data.time_wait_allocate.groupby('runNumber'):
        timeWaitAllocate_mean = calculate_mean(group['timePoint'], group['timeWaitAllocate'])
        timeWaitAllocate_std_dev = calculate_std_dev(group['timePoint'], group['timeWaitAllocate'], timeWaitAllocate_mean)
        mean_stddev_stats.timeWaitAllocate_mean.append(timeWaitAllocate_mean)
        mean_stddev_stats.timeWaitAllocate_std_dev.append(timeWaitAllocate_std_dev)

    dt = pd.DataFrame(attr.asdict(mean_stddev_stats))
    dt['params_index'] = index
    mean_stddev_stats_list.append(dt)


# In[48]:


rename_dict = {
    'diskLoad': 'Завантаження дисків',
    'ioChannelLoad': 'Завантаження каналу введення-виведення',
    'processorsLoad': 'Завантаження процесорів',
    'totalWaitAllocate': "Кількість завдань в очікуванні пам'яті",
    'useOfPage': 'Кількість зайнятих сторінок',
    'timeInSystem': 'Час завдання в системі',
    'timeWaitAllocate': "Час виділення пам'яті",
}


def split_into_means_and_stddevs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = pd.DataFrame()
    stddevs = pd.DataFrame()
    for name in data.columns:
        short_name = name.split('_')[0]
        if name.endswith('mean'):
            means[short_name] = data[name]
        else:
            stddevs[short_name] = data[name]

    means['Індекс набору параметрів'] = data['params_index']
    stddevs['Індекс набору параметрів'] = data['params_index']
    means.rename(columns=rename_dict, inplace=True)
    stddevs.rename(columns=rename_dict, inplace=True)

    column_names = means.columns.tolist()
    column_names = [column_names[-1]] + column_names[:-1]

    means = means[column_names]
    stddevs = stddevs[column_names]

    return means, stddevs


# In[49]:


mean_stddev_stats_data_frame = pd.concat(mean_stddev_stats_list, ignore_index=True)
mean_stats_data_frame, stddev_stats_data_frame = split_into_means_and_stddevs(mean_stddev_stats_data_frame)


# In[50]:


mean_stats_data_frame.to_csv(verification_res_dir_path / 'mean_stats_data_frame.csv', index=False)
mean_stats_data_frame


# In[51]:


stddev_stats_data_frame.to_csv(verification_res_dir_path / 'stddev_stats_data_frame.csv', index=False)
stddev_stats_data_frame


# In[52]:


global_mean_stddev_list: list[pd.DataFrame] = []
mean_stddev_stats_relative_mean_list: list[pd.DataFrame] = []

for i, mean_stddev_stats in mean_stddev_stats_data_frame.groupby('params_index'):
    means = mean_stddev_stats.mean()
    global_mean_stddev_list.append(means)
    mean_stddev_stats_relative_mean = ((mean_stddev_stats - means).abs() * 100) / means
    mean_stddev_stats_relative_mean.fillna(0, inplace=True)
    mean_stddev_stats_relative_mean['params_index'] = i
    mean_stddev_stats_relative_mean_list.append(mean_stddev_stats_relative_mean)


# In[53]:


global_mean_data_frame = pd.DataFrame()
global_std_dev_data_frame = pd.DataFrame()
for name in global_mean_stddev_data_frame.columns:
    short_name = name.split('_')[0]
    if name.endswith('mean'):
        global_mean_data_frame[short_name] = global_mean_stddev_data_frame[name]
    else:
        global_std_dev_data_frame[short_name] = global_mean_stddev_data_frame[name]

global_mean_data_frame.rename(columns=rename_dict, inplace=True)
global_std_dev_data_frame.rename(columns=rename_dict, inplace=True)


# In[54]:


global_mean_data_frame


# In[55]:


global_mean_data_frame.to_csv(verification_res_dir_path / 'global_mean_data_frame.csv', index=True, index_label='Індекс набору параметрів')


# In[56]:


global_std_dev_data_frame


# In[57]:


global_std_dev_data_frame.to_csv(verification_res_dir_path / 'global_std_dev_data_frame.csv', index=True, index_label='Індекс набору параметрів')


# In[58]:


mean_stddev_stats_relative_mean_data_frame = pd.concat(mean_stddev_stats_relative_mean_list, ignore_index=True)
mean_stats_relative_mean_data_frame, stddev_stats_relative_mean_data_frame = split_into_means_and_stddevs(mean_stddev_stats_relative_mean_data_frame)


# In[59]:


mean_stats_relative_mean_data_frame.to_csv(verification_res_dir_path / 'mean_stats_relative_mean_data_frame.csv', index=False)
mean_stats_relative_mean_data_frame


# In[60]:


stddev_stats_relative_mean_data_frame.to_csv(verification_res_dir_path / 'stddev_stats_relative_mean_data_frame.csv', index=False)
stddev_stats_relative_mean_data_frame

