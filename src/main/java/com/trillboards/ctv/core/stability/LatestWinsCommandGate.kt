package com.trillboards.ctv.core.stability

/**
 * Coalesces bursty commands where only the newest payload should win.
 *
 * A common case is sensing-profile rotation after a reconnect, where the socket
 * can replay multiple stale profile updates. This gate keeps at most one
 * pending command and rejects anything older than the most recently applied
 * command.
 */
class LatestWinsCommandGate<T>(
    private val orderKeyOf: (T) -> Long
) {

    enum class OfferStatus {
        ENQUEUED,
        REPLACED_PENDING,
        STALE_ALREADY_APPLIED,
        STALE_PENDING
    }

    data class OfferResult<T>(
        val status: OfferStatus,
        val replacedCommand: T? = null
    )

    private val lock = Any()
    private var pending: T? = null
    private var lastAppliedOrderKey = Long.MIN_VALUE
    private var inFlightOrderKey = Long.MIN_VALUE

    fun offer(command: T): OfferResult<T> {
        synchronized(lock) {
            val commandOrderKey = orderKeyOf(command)
            val currentFloor = maxOf(lastAppliedOrderKey, inFlightOrderKey)
            if (commandOrderKey < currentFloor) {
                return OfferResult(status = OfferStatus.STALE_ALREADY_APPLIED)
            }

            val currentPending = pending
            if (currentPending == null) {
                pending = command
                return OfferResult(status = OfferStatus.ENQUEUED)
            }

            return if (commandOrderKey >= orderKeyOf(currentPending)) {
                pending = command
                OfferResult(
                    status = OfferStatus.REPLACED_PENDING,
                    replacedCommand = currentPending
                )
            } else {
                OfferResult(status = OfferStatus.STALE_PENDING)
            }
        }
    }

    fun takeNext(): T? {
        synchronized(lock) {
            val next = pending
            pending = null
            if (next != null) {
                inFlightOrderKey = maxOf(inFlightOrderKey, orderKeyOf(next))
            }
            return next
        }
    }

    fun hasPending(): Boolean {
        synchronized(lock) {
            return pending != null
        }
    }

    fun markApplied(command: T) {
        synchronized(lock) {
            val commandOrderKey = orderKeyOf(command)
            lastAppliedOrderKey = maxOf(lastAppliedOrderKey, commandOrderKey)
            if (commandOrderKey >= inFlightOrderKey) {
                inFlightOrderKey = Long.MIN_VALUE
            }
        }
    }
}
