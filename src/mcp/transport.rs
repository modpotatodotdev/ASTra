use std::io::{self, BufRead, Write};

use anyhow::Result;

/// Maximum allowed size of an incoming message line in bytes (10 MiB).
/// Prevents memory exhaustion from a single maliciously large message.
const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

/// Read a single line up to `max_len` bytes (including trailing newline).
/// Returns:
/// - `Ok(None)` on EOF before any byte is read
/// - `Ok(Some(String))` with the raw line bytes decoded as UTF-8
fn read_line_limited<R: BufRead>(reader: &mut R, max_len: usize) -> Result<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }
        if let Some(newline_pos) = available.iter().position(|&b| b == b'\n') {
            if bytes.len() + newline_pos + 1 > max_len {
                anyhow::bail!("Incoming message exceeds maximum allowed size {}", max_len);
            }
            bytes.extend_from_slice(&available[..=newline_pos]);
            reader.consume(newline_pos + 1);
            break;
        }
        if bytes.len() + available.len() > max_len {
            anyhow::bail!("Incoming message exceeds maximum allowed size {}", max_len);
        }
        bytes.extend_from_slice(available);
        let consumed = available.len();
        reader.consume(consumed);
    }

    let line = String::from_utf8(bytes)
        .map_err(|e| anyhow::anyhow!("Incoming message is not valid UTF-8: {}", e))?;
    Ok(Some(line))
}

/// Read a single newline-delimited JSON-RPC message from stdin.
///
/// The MCP stdio transport sends one compact JSON object per line (`\n`-terminated,
/// `\r\n` accepted).  This matches the `@modelcontextprotocol/sdk` implementation used
/// by standard MCP clients such as opencode and Claude Desktop.
pub(crate) fn read_message(stdin: &io::Stdin) -> Result<Option<String>> {
    let mut lock = stdin.lock();
    loop {
        let Some(line) = read_line_limited(&mut lock, MAX_MESSAGE_SIZE)? else {
            return Ok(None); // EOF
        };

        // Strip trailing \r\n or \n
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            // Blank line – ignore and keep reading.
            continue;
        }
        return Ok(Some(trimmed.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::read_line_limited;

    #[test]
    fn limited_reader_rejects_oversized_line_before_allocating_whole_line() {
        let input = "12345\n";
        let mut cursor = Cursor::new(input.as_bytes());
        let err = read_line_limited(&mut cursor, 4).expect_err("should reject over limit");
        assert!(err.to_string().contains("maximum allowed size"));
    }

    #[test]
    fn limited_reader_allows_blank_line_for_caller_to_skip() {
        let input = "\n";
        let mut cursor = Cursor::new(input.as_bytes());
        let line = read_line_limited(&mut cursor, 16).unwrap().unwrap();
        assert_eq!(line, "\n");
    }
}

/// Write a single newline-delimited JSON-RPC message to stdout.
///
/// Serialises as `{json}\n`, matching the MCP stdio transport framing.
pub(crate) fn write_message(stdout: &mut io::Stdout, message: &str) -> Result<()> {
    stdout.write_all(message.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}
