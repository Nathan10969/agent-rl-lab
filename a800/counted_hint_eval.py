"""Compare a counted no-effect reminder on four previously failed ALFWorld cases."""
import copy
import json

from feedback_hint_eval import main


COUNTED_REMINDER = ('Reminder: the exact action {action} has received "Nothing happens." '
                    '{count} times in this episode.')


def add_counted_reminders(messages):
    """Count identical no-effect actions in the visible history without changing it."""
    if len(messages) < 41 or len(messages) % 2 != 1:
        raise ValueError("expected the initial prompt and at least 20 complete action/feedback pairs")
    if any(message.get("role") != ("assistant" if index % 2 else "user")
           for index, message in enumerate(messages)):
        raise ValueError("unexpected conversation role")

    result = copy.deepcopy(messages)
    counts = {}
    steps = []
    for index in range(2, len(messages), 2):
        action = messages[index - 1]["content"]
        feedback = messages[index]["content"]
        if feedback.strip() != "Nothing happens.":
            continue
        counts[action] = counts.get(action, 0) + 1
        if index // 2 >= 20 and counts[action] >= 2:
            result[index]["content"] = feedback + "\n" + COUNTED_REMINDER.format(
                action=json.dumps(action, ensure_ascii=False), count=counts[action])
            steps.append(index // 2)
    return result, steps


if __name__ == "__main__":
    main(reminder_transform=add_counted_reminders, reminder_description=COUNTED_REMINDER,
         reminder_mode="counted_no_effect")
