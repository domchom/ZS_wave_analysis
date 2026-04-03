"""
Tests for housekeeping_functions.py.

All inputs are constructed inline — no external assets needed.
Covers: get_channel_combos, match_group_to_file, check_frame_interval,
group_name_error_check.
"""
import math
import pytest
from waveanalysis.housekeeping.housekeeping_functions import (
    get_channel_combos,
    match_group_to_file,
    check_frame_interval,
    group_name_error_check,
)


# ── get_channel_combos ────────────────────────────────────────────────────────

def test_get_channel_combos_one_channel():
    assert get_channel_combos(1) == []

def test_get_channel_combos_two_channels():
    assert get_channel_combos(2) == [[0, 1]]

def test_get_channel_combos_three_channels():
    assert get_channel_combos(3) == [[0, 1], [0, 2], [1, 2]]

def test_get_channel_combos_four_channels():
    result = get_channel_combos(4)
    assert result == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

def test_get_channel_combos_no_duplicates():
    result = get_channel_combos(4)
    # each pair should appear exactly once
    assert len(result) == len(set(map(tuple, result)))


# ── match_group_to_file ───────────────────────────────────────────────────────

def test_match_group_to_file_match():
    assert match_group_to_file('1_Group1_data', ['Group1', 'Group2']) == 'Group1'

def test_match_group_to_file_second_group():
    assert match_group_to_file('1_Group2_data', ['Group1', 'Group2']) == 'Group2'

def test_match_group_to_file_no_match_returns_none():
    assert match_group_to_file('1_Control_data', ['Group1', 'Group2']) is None

def test_match_group_to_file_empty_group_list_returns_none():
    assert match_group_to_file('1_Group1_data', ['']) is None


# ── check_frame_interval ──────────────────────────────────────────────────────

def test_check_frame_interval_valid_passthrough():
    log = {'Errors': []}
    assert check_frame_interval(0.5, log, 'test.tif') == 0.5

def test_check_frame_interval_zero_returns_one():
    log = {'Errors': []}
    assert check_frame_interval(0, log, 'test.tif') == 1

def test_check_frame_interval_none_returns_one():
    log = {'Errors': []}
    assert check_frame_interval(None, log, 'test.tif') == 1

def test_check_frame_interval_nan_returns_one():
    log = {'Errors': []}
    assert check_frame_interval(float('nan'), log, 'test.tif') == 1

def test_check_frame_interval_invalid_logs_error():
    log = {'Errors': []}
    check_frame_interval(0, log, 'test.tif')
    assert len(log['Errors']) == 1

def test_check_frame_interval_valid_does_not_log_error():
    log = {'Errors': []}
    check_frame_interval(0.5, log, 'test.tif')
    assert len(log['Errors']) == 0


# ── group_name_error_check ────────────────────────────────────────────────────

def test_group_name_error_check_unmatched_file_logged():
    log = {'Group Matching Errors': []}
    group_name_error_check(['Group1_file.tif', 'nogroup.tif'], ['Group1'], log)
    assert any('nogroup.tif' in e for e in log['Group Matching Errors'])

def test_group_name_error_check_all_matched_no_errors():
    log = {'Group Matching Errors': []}
    group_name_error_check(['Group1_file.tif', 'Group2_file.tif'], ['Group1', 'Group2'], log)
    assert log['Group Matching Errors'] == []

def test_group_name_error_check_multiple_groups_on_one_file_exits():
    log = {'Group Matching Errors': []}
    with pytest.raises(SystemExit):
        group_name_error_check(['Group1_Group2_file.tif'], ['Group1', 'Group2'], log)

def test_group_name_error_check_missing_group_exits():
    log = {'Group Matching Errors': []}
    with pytest.raises(SystemExit):
        # Group2 is specified but no file contains it
        group_name_error_check(['Group1_file.tif'], ['Group1', 'Group2'], log)
