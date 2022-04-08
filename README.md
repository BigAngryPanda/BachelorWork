# BachelorWork

## Release build
```
cargo install --path=.
```

Result will be in
```
./target/release
```

## How to test with `nc`

1) Deploy `nc` client

```
nc -u 127.0.0.1 34254
```

2) Deploy `nc` server
```
nc -u -l 8080
```

## Cryptokey layout

First byte is LSB, latest - MSB (Little endian)