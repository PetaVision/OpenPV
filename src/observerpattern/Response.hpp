/*
 * Response.cpp
 *
 *  Created on Jan 22, 2018
 *
 *  The possible return values of the notify and response functions
 *  in the observer pattern.
 */

#ifndef RESPONSE_HPP_
#define RESPONSE_HPP_

namespace PV {

/**
 * The Response namespace defines an enum Status, and an addition operation on it.
 * Response::Status defines the return values of Observer::Respond and Subject::Notify.
 *
 * A return value of SUCCESS means that the object had not yet completed the task defined
 * by the message(s) on entry, but has now done so.
 *
 * A return value of NO_ACTION means that either the object had nothing to do in response
 * to the message(s) or had already done it at the time the method was called.
 *
 * A return value of PARTIAL means that the object has not yet completed the task,
 * but made progress toward doing so during the call.
 *
 * A return value of POSTPONE means that the object has not completed the task, and cannot
 * make progress until one or more events outside the object's control has occurred.
 */
namespace Response {

enum Status { SUCCESS, NO_ACTION, PARTIAL, POSTPONE };

/**
 * The addition operator for the Response::Status type. The rationale is that if A and B
 * are two Response::Status variables, the value of A+B is the status of a container
 * with two components, one of which returned A and the other returned B.
 *
 * For example, SUCCESS + POSTPONE == PARTIAL. One of the components did its task but
 * the other had to wait, so the container did some but not all of the task required of it.
 */
Status operator+(Status const &a, Status const &b);

/**
 * A convenience method to test if a status is either SUCCESS or NO_ACTION. In either
 * case, the message that generated this status would not need to be resent.
 */
static bool completed(Status &a) { return a == SUCCESS or a == NO_ACTION; }

} // namespace PV

} // namespace PV

#endif // RESPONSE_HPP_
