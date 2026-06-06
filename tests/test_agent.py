import unittest

from app.agent import MedicalTriageAgent
from app.database import DEFAULT_DB_PATH, init_db


class MedicalTriageAgentTest(unittest.TestCase):
    def setUp(self):
        init_db(DEFAULT_DB_PATH)
        self.agent = MedicalTriageAgent(DEFAULT_DB_PATH, use_llm=False, use_vector=False)

    def test_chest_pain_goes_to_emergency(self):
        result = self.agent.ask("胸痛半小时，出汗，左肩也疼，还有点呼吸困难")
        self.assertIn("急诊科", result.answer)
        self.assertIn("急症", result.answer)

    def test_uti_retrieval(self):
        result = self.agent.ask("最近尿频尿急尿痛，有一点血尿")
        self.assertIn("泌尿外科", result.answer)
        self.assertIn("尿路感染", result.answer)

    def test_common_cold_does_not_go_to_emergency(self):
        result = self.agent.ask("鼻塞流涕打喷嚏，轻微咳嗽，低热一天")
        self.assertIn("可先门诊/线上问诊评估", result.answer)
        self.assertIn("呼吸内科", result.answer)
        self.assertNotIn("建议立即前往急诊科", result.answer)

    def test_single_chest_discomfort_is_not_automatic_emergency(self):
        result = self.agent.ask("饭后胸口灼热反酸嗳气，平卧后更明显")
        self.assertIn("可先门诊/线上问诊评估", result.answer)
        self.assertIn("消化内科", result.answer)
        self.assertNotIn("建议立即前往急诊科", result.answer)

    def test_response_time_is_recorded(self):
        result = self.agent.ask("最近尿频尿急尿痛")
        self.assertGreaterEqual(result.response_time_ms, 0)

    def test_unmatched_query_does_not_fall_back_to_emergency(self):
        result = self.agent.ask("今天感觉有点说不出来的不舒服")
        self.assertIn("信息不足", result.answer)
        self.assertIn("全科/普通内科", result.answer)
        self.assertNotIn("建议立即前往急诊科", result.answer)


if __name__ == "__main__":
    unittest.main()
