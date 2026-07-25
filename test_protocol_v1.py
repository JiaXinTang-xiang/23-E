import unittest

import protocol_v1 as proto


class ProtocolV1Tests(unittest.TestCase):
    def test_crc_standard_vector(self):
        self.assertEqual(proto.crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_sequence_wraps(self):
        sequence = proto.SequenceCounter(254)
        self.assertEqual(
            [sequence.next() for _ in range(4)], [254, 255, 0, 1])

    def test_rectangle_round_trip(self):
        points = [(10, 20), (630, 20), (630, 460), (10, 460)]
        frame = proto.pack_frame(
            proto.CMD_OUTER_RECT,
            proto.pack_rect_payload(points),
            flags=proto.FLAG_DATA_VALID,
            seq=7,
        )
        decoded = proto.unpack_frame(frame)

        self.assertEqual(len(frame), 29)
        self.assertEqual(decoded["cmd"], proto.CMD_OUTER_RECT)
        self.assertEqual(decoded["seq"], 7)
        self.assertEqual(
            proto.unpack_rect_payload(decoded["payload"]), points)

    def test_laser_round_trip(self):
        frame = proto.pack_frame(
            proto.CMD_LASER_POINT,
            proto.pack_laser_payload(320, 240, color=1, confidence=100),
            flags=proto.FLAG_DATA_VALID,
        )
        decoded = proto.unpack_frame(frame)

        self.assertEqual(len(frame), 18)
        self.assertEqual(
            proto.unpack_laser_payload(decoded["payload"]),
            {"color": 1, "x": 320, "y": 240, "confidence": 100},
        )

    def test_parser_handles_noise_split_and_concatenated_frames(self):
        rect = proto.pack_frame(
            proto.CMD_OUTER_RECT,
            proto.pack_rect_payload([(1, 2), (3, 4), (5, 6), (7, 8)]),
        )
        laser = proto.pack_frame(
            proto.CMD_LASER_POINT, proto.pack_laser_payload(100, 200))
        stream = b"noise" + rect + laser
        parser = proto.ProtocolParser()
        decoded = []

        for size in (1, 2, 7, 3, 20, 1000):
            decoded.extend(parser.feed(stream[:size]))
            stream = stream[size:]
        decoded.extend(parser.feed(stream))

        self.assertEqual(
            [frame["cmd"] for frame in decoded],
            [proto.CMD_OUTER_RECT, proto.CMD_LASER_POINT],
        )

    def test_parser_rejects_bad_crc_and_recovers(self):
        bad = bytearray(proto.pack_frame(
            proto.CMD_LASER_POINT, proto.pack_laser_payload(100, 200)))
        bad[12] ^= 0x01
        good = proto.pack_frame(
            proto.CMD_HEARTBEAT, proto.pack_heartbeat_payload(1000))

        decoded = proto.ProtocolParser().feed(bad + good)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["cmd"], proto.CMD_HEARTBEAT)


if __name__ == "__main__":
    unittest.main()
