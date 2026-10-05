"""Protocol tests use no socket and never enter an official test."""
import io
import json
import time
import unittest
from urllib.error import URLError
from q3_environment import Environment
from q3_simulator_client import Client, ProtocolError, public_feedback, run_session


class MockProtocol:
    def __init__(self, seed=1040000):
        self.env = Environment(seed); self.cache = {}; self.calls = []; self.drop = False
    def __call__(self, url, data, timeout):
        body = json.loads(data); path = '/'+url.rsplit('/',1)[1]
        self.calls.append((path,data))
        assert set(body) == ({'arena_id','robot_id','request_id','position','channel'} if path in ('/measure','/clear') else {'arena_id','robot_id','request_id'})
        assert body['arena_id'] == 'default' and body['robot_id'] == 'test-team'
        key = body['request_id']
        if key in self.cache:
            old_path, old_data, response = self.cache[key]
            return (200,response) if (path,data)==(old_path,old_data) else (409,dict(accepted=False,virtual_time_s=0))
        if path == '/enter':
            response = dict(accepted=True,virtual_time_s=0,remaining_real_duration_s=1200)
        elif path == '/exit':
            assert self.env.successes == self.env.n
            response = dict(accepted=True,virtual_time_s=self.env.time,exit_reason='user_exit')
        else:
            assert set(body['position']) == {'x','y'}
            result = self.env.act(path[1:],body['channel'],[body['position']['x'],body['position']['y']])
            response = dict(accepted=True,virtual_time_s=result['virtual_time_s'])
            response['measure_result' if path == '/measure' else 'clear_result'] = result['kind']
            if 'angle' in result: response['svd_deg'] = result['angle']
        self.cache[key] = path,data,response
        if self.drop:
            self.drop = False
            raise URLError('Response lost after execution')
        return 200,response


class ClientTests(unittest.TestCase):
    def test_retry_reuses_exact_bytes_and_does_not_double_execute(self):
        server = MockProtocol(); client = Client('test-team',transport=server,retry_delay=0)
        client.post('/enter'); server.drop = True
        client.post('/measure',dict(channel=1,position=[300,400]))
        self.assertEqual(server.env.measurements,1)
        self.assertEqual(server.calls[-1],server.calls[-2])

    def test_rejection_does_not_reset_last_accepted_clock(self):
        client = Client('test-team',transport=lambda *a:(200,dict(accepted=False,virtual_time_s=0)))
        client.virtual_time_s = 123.5
        with self.assertRaises(ProtocolError): client.post('/measure',dict(channel=1,position=[0,0]))
        self.assertEqual(client.virtual_time_s,123.5)

    def test_http_status_is_checked_even_if_body_claims_accepted(self):
        client = Client('test-team',transport=lambda *a:(409,dict(accepted=True,virtual_time_s=0)))
        with self.assertRaises(ProtocolError): client.post('/enter')

    def test_elapsed_actual_deadline_prevents_transport(self):
        calls=[]; client=Client('test-team',transport=lambda *a:calls.append(a))
        client.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): client.post('/enter')
        self.assertFalse(calls)

    def test_direction_conversion_requires_real_reading(self):
        with self.assertRaises(ProtocolError): public_feedback(dict(action='measure'),dict(measure_result='direction',virtual_time_s=5))
        self.assertEqual(public_feedback(dict(action='clear'),dict(clear_result='success',virtual_time_s=10))['kind'],'success')

    def test_invalid_coordinates_never_reach_transport(self):
        calls=[]; client=Client('test-team',transport=lambda *a:calls.append(a))
        with self.assertRaises(ValueError): client.post('/measure',dict(channel=1,position=[float('nan'),0]))
        self.assertFalse(calls)

    def test_full_frozen_policy_through_serial_protocol(self):
        server=MockProtocol(); log=io.StringIO(); client=Client('test-team',transport=server,log=log)
        result=run_session(client)
        self.assertEqual(result['status'],'completed_with_certificate')
        self.assertEqual(result['cleared'],server.env.n)
        self.assertAlmostEqual(result['virtual_s'],server.env.time)
        self.assertIsNone(result['source_count'])
        self.assertIsNone(result['clear_fraction'])
        self.assertEqual(server.calls[-1][0],'/exit')
        self.assertIn('policy_state',log.getvalue())


if __name__ == '__main__': unittest.main()
