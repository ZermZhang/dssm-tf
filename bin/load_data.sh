#!/bin/bash
# The script load processed data.
# Usage:
#       1. bash load_process_data.sh train  # default to load yesterday train data
#       2. bash load_process_data.sh train 20180110  # one param $1 to load certain date data
#       3. bash load_process_data.sh train 20180110 20180120  # two params $1 $2 to load data from $1 to $2

#set -e

dt=`date -d "yesterday" +%Y%m%d`
end_dt=$dt

data_type=$1
if [ $# -eq 2 ]; then
    dt=$2
    end_dt=$dt
elif [ $# -eq 3 ]; then
    dt=$2
    end_dt=$3
fi

from_s3_train_dir=s3://wangyungan/zhangzhen/sort/data/fine_data_v1.1/${data_type}
to_local_temp_dir=./data/temp_data
to_local_mid_dir=./data/mid_data
to_local_dir=./data/${data_type}

if [ ! -d "$to_local_dir" ]; then
    mkdir -p "$to_local_dir" && echo "Already make local dir: $to_local_dir"
    mkdir -p $to_local_temp_dir && rm -rf $to_local_temp_dir/*  && echo "Local temp data dir is cleaned."
fi

function test_finish() {
    t_dt=$1
    t_path=${from_s3_train_dir}/$t_dt
    exits=$(aws s3 ls ${t_path}/_SUCCESS)
    if [ ! -n "$exits" ]; then
        echo "Data:$t_path is not exits."
        return 1
    else
        echo "Data:$t_path was finished."
        return 0
    fi
}

function load_data() {
    dt=$1
    data_type=$2
    input_path=${from_s3_train_dir}/${dt}
    output=${to_local_dir}/${dt}
    echo "Start loading data from s3: ${input_path}."
    rm -rf $to_local_temp_dir/*
    aws s3 sync ${input_path} $to_local_temp_dir
    # todo: to do it better, using cat is ugly
    # cat $to_local_temp_dir/* | awk -F '\t' '{if($17 == "page_real_class" || $17=="page_select_class" || $17=="page_virtual_class" || $17=="page_goods_group") {print}}' | awk -F '\t' '{if($41=="0") {print}}' | awk -F '\t' '{if($1=="3" || $1=="0") {print}}' > ${output}

    if [ "${data_type}"x = "train"x ]; then
        echo "sample train data"
        cat ${to_local_temp_dir}/* | awk 'BEGIN{FS="\t"; OFS="\t"} {if($14 == "page_real_class" || $14=="page_select_class" || $14=="page_virtual_class" || $14=="page_goods_group") {print}}' | awk 'BEGIN{FS="\t"; OFS="\t"} {if($2 == "us" || $2=="iosshus" || $2=="pwus" || $2=="andshus") {print}}' | awk 'BEGIN{FS="\t"; OFS="\t"} {if($1!="0"){gsub($1,"1",$1)}; print}' > ${to_local_mid_dir}/${dt}
        # sh -c "python ./tools/sample_data.py ${to_local_temp_dir} ${to_local_mid_dir}/$dt ${data_type}"
        sh -c "cat ${to_local_mid_dir}/* | shuf > ${to_local_dir}/${dt}"
    fi
    if [ "${data_type}"x = "test"x ]; then
        echo "sample test data"
        cat ${to_local_temp_dir}/* | awk 'BEGIN{FS="\t"; OFS="\t"} {if($14 == "page_real_class" || $14=="page_select_class" || $14=="page_virtual_class" || $14=="page_goods_group") {print}}' | awk 'BEGIN{FS="\t"; OFS="\t"} {if($2 == "us" || $2=="iosshus" || $2=="pwus" || $2=="andshus") {print}}' | awk 'BEGIN{FS="\t"; OFS="\t"} {if($1!="0"){gsub($1,"1",$1)}; print}' > ${to_local_dir}/${dt}
        # sh -c "python ./tools/sample_data.py ${to_local_temp_dir} ${to_local_dir}/$dt ${data_type}"
    fi


    rm -rf $to_local_temp_dir/*
    echo "Finish loading data to local: ${output}."
}

max_times=30
try_time=0
if [ $dt -ne $end_dt ]; then
    cur_dt=$dt
    while [ $cur_dt -le $end_dt ]
  do
        test_finish $cur_dt
        ret=$?
        while [ $ret -ne 0 ]
        do
            echo 'sleep 10min'
            sleep 10m
            test_finish $cur_dt
            ret=$?
            let try_time=$try_time+1
            if [ $try_time -gt $max_times ];then
                echo "data of [$cur_dt]'s try times over than $max_times"
                break
            fi
        done
        if [ $try_time -le $max_times ];then
            load_data $cur_dt ${data_type}
        fi
        cur_dt=`date -d "$cur_dt 1days" +%Y%m%d`
    done
else
    cur_dt=$dt
    test_finish $cur_dt
    ret=$?
    while [ $ret -ne 0 ]
    do
        echo 'sleep 10min'
        sleep 10m
        test_finish $cur_dt
        ret=$?
        let try_time=$try_time+1
        if [ $try_time -gt $max_times ]; then
            echo "Try times over than $max_times"
            break
        fi
    done
    if [ $try_time -le $max_times ]; then
        load_data $dt ${data_type}
        echo "Done! see data in ${to_local_dir}."
    fi
fi

if [ $try_time -gt $max_times ];then
    exit 1
fi

