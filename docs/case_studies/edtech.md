# Case Study IV — Educational Technology

## Turning Student Misconceptions into Curriculum

An edtech startup builds a tutoring LLM for high school physics. Factual accuracy: 91%. The 9% wrong answers cluster around specific conceptual boundaries: mass vs. weight, direction of friction in rolling motion, sign convention in thermodynamics. The model confabulates at the same boundaries as human students — because it was trained on human-written text that over-represents the same misconceptions.

## The PHANTASM intervention

HGT traces every answer over three months of student interactions.

- HGT identifies 23 recurring `knowledge_gap` patterns — positions where gradient norms spike consistently across diverse queries
- Each gap pattern corresponds to a specific conceptual boundary in the physics curriculum
- The patterns match exactly with the most-failed items on end-of-semester exams

The startup rebuilds curriculum scaffolding around the 23 knowledge-gap patterns. Targeted micro-modules are created for each.

## Outcome

Student performance on previously-failed exam items: **+34%** in the following semester.

## The key reversal

The model's hallucinations were not errors in the tutoring system. They were a perfect map of where the curriculum needed reinforcement — a map that would have taken human designers years and thousands of student failures to construct.

HGT produced it in three months, for free, from failures.
