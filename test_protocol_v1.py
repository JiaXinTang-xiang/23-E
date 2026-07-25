import struct
import unittest

import protocol_commands as commands
import protocol_v1 as proto


class ProtocolV1Tests(unittest.TestCase):
    def test_crc_standard_vector(self):
        self.assertEqual(proto.crc16(b"123456789"), 0x29B1)

    def test_rectangle_round_trip(self):
        outer = (10, 20, 630, 20, 630, 460, 10, 460)
        inner = (20, 30, 620, 30, 620, 450, 20, 450)
        payload = struct.pack(
            commands.FMT_A4_TARGET, 640, 480, *outer, *inner)
        frame = proto.make_frame(
            commands.CMD_A4_TARGET,
            payload,
            flags=proto.FLAG_DATA_VALID,
            seq=7,
        )
        decoded = proto.decode_frame(frame)

        self.assertEqual(len(frame), 48)
        self.assertEqual(decoded["cmd"], commands.CMD_A4_TARGET)
        self.assertEqual(decoded["seq"], 7)
        self.assertEqual(
            struct.unpack(commands.FMT_A4_TARGET, decoded["payload"]),
            (640, 480) + outer + inner,
        )

    def test_laser_round_trip(self):
        payload = struct.pack(commands.FMT_RED_LASER, 320, 240)
        frame = proto.make_frame(
            commands.CMD_RED_LASER,
            payload,
            flags=proto.FLAG_DATA_VALID,
            seq=8,
        )
        decoded = proto.decode_frame(frame)

        self.assertEqual(len(frame), 16)
        self.assertEqual(
            struct.unpack(commands.FMT_RED_LASER, decoded["payload"]),
            (320, 240),
        )

    def test_stream_parser(self):
        rect = proto.make_frame(
            commands.CMD_A4_TARGET,
            struct.pack(
                commands.FMT_A4_TARGET,
                640, 480,
                1, 2, 3, 4, 5, 6, 7, 8,
                11, 12, 13, 14, 15, 16, 17, 18,
            ),
            seq=10,
        )
        laser = proto.make_frame(
            commands.CMD_RED_LASER,
            struct.pack(commands.FMT_RED_LASER, 100, 200),
            seq=11,
        )
        stream = b"noise" + rect + laser
        buffer = bytearray()
        decoded = []

        for size in (1, 2, 7, 3, 20, 1000):
            decoded.extend(proto.parse_frames(buffer, stream[:size]))
            stream = stream[size:]
        decoded.extend(proto.parse_frames(buffer, stream))

        self.assertEqual(
            [frame["cmd"] for frame in decoded],
            [commands.CMD_A4_TARGET, commands.CMD_RED_LASER],
        )

    def test_bad_crc_is_discarded_and_parser_recovers(self):
        bad = bytearray(proto.make_frame(
            commands.CMD_RED_LASER,
            struct.pack(commands.FMT_RED_LASER, 100, 200),
            seq=12,
        ))
        bad[12] ^= 0x01
        good = proto.make_frame(
            commands.CMD_HEARTBEAT,
            struct.pack(commands.FMT_HEARTBEAT, 1000, 0, 0),
            seq=13,
        )

        decoded = proto.parse_frames(bytearray(), bad + good)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["cmd"], commands.CMD_HEARTBEAT)

    def test_ack_reuses_request_sequence_and_command(self):
        request_frame = proto.make_frame(
            commands.CMD_SET_MODE,
            struct.pack(commands.FMT_SET_MODE, 2, 1),
            flags=proto.FLAG_NEED_ACK,
            src=proto.DEVICE_MAIN_MCU,
            dst=proto.DEVICE_VISION,
            seq=25,
        )
        request = proto.decode_frame(request_frame)
        ack = proto.decode_frame(proto.make_ack(request, result=0))

        self.assertEqual(ack["seq"], 25)
        self.assertEqual(ack["cmd"], commands.CMD_SET_MODE)
        self.assertEqual(ack["src"], proto.DEVICE_VISION)
        self.assertEqual(ack["dst"], proto.DEVICE_MAIN_MCU)
        self.assertEqual(ack["flags"], proto.FLAG_IS_ACK)
        self.assertEqual(struct.unpack(commands.FMT_ACK, ack["payload"]), (0,))


if __name__ == "__main__":
    unittest.main()
