from dify_trace_futureagi.ids import StableIdGenerator, next_ids, stable_span_id, stable_trace_id


def test_trace_id_is_stable_and_128_bit():
    assert stable_trace_id("run-1") == stable_trace_id("run-1")
    assert stable_trace_id("run-1") != stable_trace_id("run-2")
    assert 0 < stable_trace_id("run-1") < 1 << 128


def test_span_id_is_stable_and_64_bit():
    assert stable_span_id("run-1", "workflow", "node-1") == stable_span_id("run-1", "workflow", "node-1")
    assert stable_span_id("run-1", "workflow", "node-1") != stable_span_id("run-1", "workflow", "node-2")
    assert 0 < stable_span_id("run-1", "workflow", "node-1") < 1 << 64


def test_workflow_and_message_traces_of_one_turn_do_not_share_a_span_id():
    """Both carry the message id as a span id; with an external trace id they also share the trace id."""
    assert stable_span_id("ext-1", "chatflow_run-1", "message-1") != stable_span_id("ext-1", "message", "message-1")


def test_generator_uses_the_ids_set_for_the_block_and_random_ids_outside():
    generator = StableIdGenerator()
    with next_ids(7, 9):
        assert generator.generate_trace_id() == 7
        assert generator.generate_span_id() == 9
    assert len({generator.generate_span_id() for _ in range(3)}) == 3
    assert len({generator.generate_trace_id() for _ in range(3)}) == 3
