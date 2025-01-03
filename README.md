# EconELO

The **EconELO** module is a comprehensive economy and points management system for Discord servers. It features a federal reserve system, daily rewards, casino games, user statistics tracking, and an advanced leveling system.

## Features

- Points-based economy with federal reserve management
- Multi-tier role rewards and status levels
- Dynamic tax system based on transaction amounts
- Anti-abuse protection and rate limiting
- Comprehensive logging system
- Debt management with interest rates
- Daily/weekly/monthly/seasonal/yearly claim rewards
- Role-based reward multipliers
- Message quality rewards with entropy analysis
- Reaction rewards with special emoji bonuses
- Invite rewards with stay requirements
- Newbie task completion bonuses
- Coin Flip (1.5x-3x multiplier options)
- Dice Roll (2x for doubles, 3x for double sixes)
- Number Guessing (1-100 range, dynamic multipliers)
- Rock Paper Scissors (PvP or vs House)
- Experience-based leveling
- Unlockable titles and roles
- Skill progression (Leadership, Helpfulness, Creativity, Engagement)
- Achievement tracking
- Weekly activity rewards

## Usage (Slash Commands)

### Claim Commands

- `/econelo claim daily`
  - Claim your daily points reward based on your status role
  - Cooldown: 24 hours
  - Points vary by status:
    - `ELECTORAL_ROLE_ID`: 15 points
    - `APPROVED_ROLE_ID`: 10 points
    - `TEMPORARY_ROLE_ID`: 5 points
  - Points are automatically adjusted for tax
  - Note: For role ID definitions, please refer to [Roles](https://github.com/kazuki388/Roles)

- `/econelo claim role <type>`
  - Claim role-based rewards at different intervals
  - Required:
    - `type`: daily | weekly | monthly | seasonal | yearly
  - Cooldowns:
    - Daily: 24 hours
    - Weekly: 7 days
    - Monthly: 30 days
    - Seasonal: 90 days
    - Yearly: 365 days
  - Rewards scale with role tier:
    - Tier 1: 50/500/2500/8000/35000 points
    - Tier 2: 40/400/2000/6000/28000 points
    - Tier 3: 35/350/1500/5000/22000 points
    - Tier 4: 30/300/1000/4000/18000 points

### Casino Commands

- `/econelo casino flip <bet> <choice> [multiplier] [opponent]`
  - Classic heads or tails betting game
  - Required:
    - `bet`: Amount to wager (minimum 1)
    - `choice`: heads | tails
  - Optional:
    - `multiplier`: safe (1.5x) | normal (2x) | risky (3x)
    - `opponent`: @user to play against (default: house)
  - Tax applies to winnings
  - House edge: 2%

- `/econelo casino dice <bet> [opponent]`
  - Roll two dice and compete for highest total
  - Required:
    - `bet`: Amount to wager (minimum 1)
  - Optional:
    - `opponent`: @user to play against (default: house)
  - Special multipliers:
    - Double sixes: 3x payout
    - Any doubles: 2x payout
    - Regular win: 1x payout
  - Tax applies to winnings

- `/econelo casino guess <bet>`
  - Guess a number between 1-100 in 5 rounds or less
  - Required:
    - `bet`: Amount to wager (minimum 1)
  - Multipliers by round:
    - Round 1: 10x payout
    - Round 2: 5x payout
    - Round 3: 2x payout
    - Round 4: 1x payout
    - Round 5: 0.5x payout
  - Hints provided after each guess
  - 30 second timeout per guess

- `/econelo casino rps <bet> <choice> [opponent]`
  - Rock, Paper, Scissors with betting
  - Required:
    - `bet`: Amount to wager (minimum 1)
    - `choice`: rock | paper | scissors
  - Optional:
    - `opponent`: @user to play against (default: house)
  - Standard RPS rules apply
  - Winner takes bet amount minus tax
  - Ties return bets to players

### View Commands

- `/econelo view profile [user]`
  - View detailed user statistics and progress
  - Optional:
    - `user`: @user to view (default: self)
  - Displays:
    - Current points balance
    - Total lifetime earnings
    - Current level and title
    - Weekly activity score
    - Message/reaction counts
    - Gambling statistics
    - Skill levels and progress
    - Achievement completion
    - Claim cooldowns

- `/econelo view leaderboard`
  - View server-wide rankings and statistics
  - Features:
    - Top 10 point balances
    - Total earnings rankings
    - Weekly activity leaders
    - Gambling profit rankings
    - Level distribution
    - Interactive navigation
    - Personal rank indicator

### Help Commands

- `/econelo help main`
  - Overview of the entire system
  - Covers:
    - Basic mechanics
    - Point sources
    - Tax system
    - Role benefits
    - Command categories
    - Getting started tips

- `/econelo help claim`
  - Detailed guide for claiming rewards
  - Explains:
    - Available claim types
    - Role requirements
    - Cooldown timers
    - Reward calculations
    - Tax implications
    - Optimal claiming strategies

- `/econelo help casino`
  - Comprehensive casino game guide
  - Details:
    - Game rules
    - Betting limits
    - House edges
    - Multiplier systems
    - Risk management
    - PvP mechanics
    - Tax calculations

- `/econelo help view`
  - Guide to viewing statistics and information
  - Covers:
    - Profile interpretation
    - Leaderboard navigation
    - Stat tracking
    - Progress monitoring
    - Achievement viewing
    - Historical data access

## Configuration

### Role

The role is divided into two main categories:

#### Status Roles

Status roles reflect a member's standing and contributions within the community.

#### Reward Roles (Servant)

Reward Roles are special distinctions earned through dedicated service to the community.

For detailed role specifications and requirements, please consult the [Roles](https://github.com/kazuki388/Roles).

### Level

The progression has three major ranks:

  - Level 1 - Novice (666,666 points)
  - Level 2 - Expert (888,888 points)
  - Level 3 - Legend (999,999 points)

### Economy

The economy is carefully balanced through several mechanisms:

- Inflation: 3% base rate with dynamic adjustments
- Decay: 1% natural decrease to prevent hoarding
- Market: 5% volatility for engaging trading
- Reward:
  - 20% bonus for consecutive wins
  - 20% penalty for excessive losses
  - 50% seasonal event boosts

### Federal Reserve

The central banking maintains economic stability:

- Reserve:
  - Initial pool: 99,999,999 points
  - Minimum reserve: 5,000,000 points
  - Emergency intervention threshold

- Economic:
  - 5% base interest rate
  - 2% inflation target
  - 1% maximum bet ratio of total reserve
  - 5% house edge in games

## Algorithm

### Anti-Abuse

- Rate
  - Message rewards: Max 1 reward per 30 seconds
  - Daily claim: Once per 24 hours
  - Casino games: 5 second cooldown between bets
  - Invite rewards: Maximum 5 invites per day

- Transaction
  - Real-time validation of all point transfers
  - Automatic flagging of suspicious patterns
  - Multi-step verification for large transactions
  - Complete audit trail with timestamps

- Economic
  - Dynamic tax rates (5-30%) based on transaction size
  - Debt prevention with strict balance checks
  - Maximum bet limits tied to Fed reserve
  - Anti-inflation mechanisms

### Message Reward

- Base Points
  - Minimum: 1 point
  - Maximum: 5 points per message
  - Daily limit: 50-150 points (randomly determined)

- Multipliers
  - Length bonus: +2 points for 100+ characters
  - Image/file attachments: +3 points
  - Link sharing: +2 points
  - High entropy (unique content): +3 points
  - Community engagement:
    - Replies: +2 points
    - Reactions received: +1 point per unique reaction
  - Quality: 1.5x for meeting all criteria

- Penalties
  - Spam detection: 50% point reduction
  - Repetitive content: 30% point reduction
  - Excessive frequency: Temporary cooldown

### Reaction Reward

- Star Reactions (⭐)
  - Base reward: 5 points
  - Daily limit: 50 points
  - Cooldown: 5 minutes between rewards

- Special Emoji Bonuses
  - 👍 : 10 points
  - 💎 : 8 points
  - 🌟 : 7 points
  - Anti-abuse: 30 minute cooldown per emoji type

### Invite Reward

- Base Rewards
  - Successful invite: 10 points
  - Stay requirement: 7 days minimum
  - Daily limit: 5 invites
  - Points refund: Full refund if invitee leaves early

- Milestone Bonuses
  - 5 successful invites: +100 points
  - 10 successful invites: +250 points
  - 25 successful invites: +1000 points

### Weekly Activity

- Point Rewards
  - Messages sent
  - Voice channel time
  - Reactions given/received
  - Bot command usage
  - Event participation

- Bonuses (Daily login streak multipliers)
  - 7 days: 1.5x
  - 30 days: 2x
  - 100 days: 3x

- Progression
  - Automatic level calculations
  - Role requirement verification
  - Title unlocks
  - Skill progression tracking
    - Creativity
    - Engagement
    - Leadership
    - All skills cap at 100
